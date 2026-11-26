"""WMP tensor logic backed by Isaac Lab; policy coordinates remain xyzw."""
from types import SimpleNamespace
import math, numpy as np, torch
from .logic import Logic
from .competition_tasks import CompetitionMixin
from .config import asset_contract
from .math import *

class Core(CompetitionMixin, Logic):
    def __init__(self,backend):
        self.backend=backend; self.cfg=backend.cfg.legacy; self.device=backend.device
        self.sim_params=SimpleNamespace(dt=backend.physics_dt)
        self.num_envs=backend.num_envs; self.num_actions=12; self.num_obs=self.num_privileged_obs=self.cfg.env.num_observations
        self.privileged_dim=self.cfg.env.privileged_dim; self.height_dim=self.cfg.env.height_dim
        self.include_history_steps=None; self.num_dof=self.num_dofs=12; self.num_bodies=backend.robot.num_bodies
        self.up_axis_idx=2; self.init_done=False
        prev=0
        for i,name in enumerate(['wave','slope','stairup','stairdown','discrete','gap','pit','tilt','crawl','roughflat']):
            end=math.ceil(self.num_envs*sum(self.cfg.terrain.terrain_proportions[:i+1])) if i<9 else self.num_envs
            setattr(self,name+'_start_idx',prev); setattr(self,name+'_end_idx',end); prev=end
        _,self.dof_names,limits,root=asset_contract(self.cfg)
        self.feet_indices=self.body_indices([self.cfg.asset.foot_name])
        self.penalised_contact_indices=self.body_indices(self.cfg.asset.penalize_contacts_on)
        self.termination_contact_indices=self.body_indices(self.cfg.asset.terminate_after_contacts_on)
        self.terrain=backend.terrain; self._parse_cfg(self.cfg); self._get_env_origins()
        if self.terrain is None:
            self.terrain_levels=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        else:
            self.height_samples=torch.tensor(self.terrain.heightsamples,device=self.device)
            self.x_edge_mask=torch.tensor(self.terrain.x_edge_mask,device=self.device)
        self.base_init_state=to_torch(self.cfg.init_state.pos+self.cfg.init_state.rot+self.cfg.init_state.lin_vel+self.cfg.init_state.ang_vel,device=self.device)
        self.obs_buf=torch.zeros(self.num_envs,self.num_obs,device=self.device); self.privileged_obs_buf=self.obs_buf.clone()
        self.rew_buf=torch.zeros(self.num_envs,device=self.device); self.reset_buf=torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
        self.time_out_buf=torch.zeros_like(self.reset_buf); self.episode_length_buf=backend.episode_length_buf
        self.dof_pos_limits=to_torch([[limits[n]['lower'],limits[n]['upper']] for n in self.dof_names],device=self.device)
        mid=self.dof_pos_limits.mean(-1); width=self.dof_pos_limits[:,1]-self.dof_pos_limits[:,0]
        self.dof_pos_limits=torch.stack((mid-width*.5*self.cfg.rewards.soft_dof_pos_limit,mid+width*.5*self.cfg.rewards.soft_dof_pos_limit),-1)
        self.dof_vel_limits=to_torch([limits[n]['velocity'] for n in self.dof_names],device=self.device)
        self.torque_limits=to_torch([limits[n]['effort'] for n in self.dof_names],device=self.device)
        backend.sync_state(self); self.init_tensors(); self._prepare_reward_function()
        self.randomized_frictions=torch_rand_float(*self.cfg.domain_rand.friction_range,(self.num_envs,1),self.device)
        self.randomized_restitutions=torch_rand_float(*self.cfg.domain_rand.restitution_range,(self.num_envs,1),self.device)
        self.randomized_added_masses=torch_rand_float(*self.cfg.domain_rand.added_mass_range,(self.num_envs,1),self.device)
        self.randomized_com_pos=torch.cat([torch_rand_float(*getattr(self.cfg.domain_rand,'com_'+a+'_pos_range'),(self.num_envs,1),self.device) for a in 'xyz'],1)
        self.global_counter=0; self.total_env_steps_counter=0
        self.reward_curriculum_coef=[s[2] for s in self.cfg.rewards.reward_curriculum_schedule]
        self.init_competition()
        self._custom_terrain_masks=False
        if self.terrain is not None and not getattr(self.cfg.terrain,'competition',False) and len(self.cfg.terrain.terrain_proportions)>10:
            kinds=[self.terrain.column_types[int(c)] for c in self.terrain_types.tolist()]
            self._custom_terrain_masks=True
            self._roughflat_mask=torch.tensor([k=='roughflat' for k in kinds],device=self.device,dtype=torch.bool)
            self._straight_mask=~self._roughflat_mask
            self._zero_heading_mask=torch.tensor([k in ('tilt','pit','tire_array','high_wall') for k in kinds],device=self.device,dtype=torch.bool)
            self._edge_stumble_mask=torch.tensor([k in ('gap','pit','tire_array','high_wall') for k in kinds],device=self.device,dtype=torch.bool)
        self.init_done=True
    def body_indices(self,patterns):
        return torch.tensor([i for p in patterns for i,n in enumerate(self.backend.robot.body_names) if p in n],device=self.device,dtype=torch.long)
    def _reset_dofs(self,ids):
        q=self.default_dof_pos*torch_rand_float(.5,1.5,(len(ids),12),self.device)
        if getattr(self,'course',None) is not None:
            mask=self.course.mask[ids]
            q[mask]=self.default_dof_pos*torch_rand_float(.95,1.05,(int(mask.sum()),12),self.device)
        self.dof_pos[ids]=q; self.dof_vel[ids]=0
        self.backend.robot.write_joint_state_to_sim(q,torch.zeros_like(q),joint_ids=self.backend.joint_ids,env_ids=ids)
    def _reset_root_states(self,ids):
        self.root_states[ids]=self.base_init_state; self.root_states[ids,:3]+=self.env_origins[ids]
        if self.custom_origins:
            extent = getattr(self.cfg.terrain, "spawn_jitter_m", 1.)
            jitter=torch_rand_float(-extent,extent,(len(ids),),self.device)
            if getattr(self,'course',None) is not None:
                jitter=torch.where(self.course.mask[ids],jitter*.025,jitter)
            self.root_states[ids,0]+=jitter
        self.root_states[ids,7:13]=torch_rand_float(-.5,.5,(len(ids),6),self.device)
        if getattr(self,'course',None) is not None:
            new_ids=ids[self.course.mask[ids]]
            self.root_states[new_ids,7:13]=0.
        self.backend.write_root(self.root_states[ids],ids)
    def _push_robots(self):
        ids=torch.arange(self.num_envs,device=self.device)
        if getattr(self,'course',None) is not None:ids=ids[~self.course.mask]
        self.root_states[ids,7:9]=torch_rand_float(-self.cfg.domain_rand.max_push_vel_xy,self.cfg.domain_rand.max_push_vel_xy,(len(ids),2),self.device)
        self.backend.robot.write_root_velocity_to_sim(self.root_states[ids,7:13],ids)
    def _post_physics_step_callback(self):
        if not getattr(self,'_custom_terrain_masks',False):
            return super()._post_physics_step_callback()
        env_ids=(self.episode_length_buf%int(self.cfg.commands.resampling_time/self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward=quat_apply(self.base_quat,self.forward_vec)
            mask=self._straight_mask
            heading=torch.atan2(forward[mask,1],forward[mask,0])
            self.commands[mask,2]=torch.clip(0.5*wrap_to_pi(self.commands[mask,3]-heading),-1.,1.)
        if self.cfg.terrain.measure_heights:
            self.measured_heights=self._get_heights()
            self.measured_forward_heights=self._get_forward_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter%self.cfg.domain_rand.push_interval==0):
            self._push_robots()
    def _resample_commands(self,env_ids):
        if not getattr(self,'_custom_terrain_masks',False):
            return super()._resample_commands(env_ids)
        self.commands[env_ids,0]=torch_rand_float(self.command_ranges["lin_vel_x"][0],self.command_ranges["lin_vel_x"][1],(len(env_ids),1),device=self.device).squeeze(1)
        self.commands[env_ids,1]=torch_rand_float(self.command_ranges["lin_vel_y"][0],self.command_ranges["lin_vel_y"][1],(len(env_ids),1),device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids,3]=torch_rand_float(self.command_ranges["heading"][0],self.command_ranges["heading"][1],(len(env_ids),1),device=self.device).squeeze(1)
        else:
            self.commands[env_ids,2]=torch_rand_float(self.command_ranges["ang_vel_yaw"][0],self.command_ranges["ang_vel_yaw"][1],(len(env_ids),1),device=self.device).squeeze(1)
        flat_env_ids=env_ids[self._roughflat_mask[env_ids]]
        if len(flat_env_ids)>0:
            self.commands[flat_env_ids,0]=torch_rand_float(self.command_ranges["flat_lin_vel_x"][0],self.command_ranges["flat_lin_vel_x"][1],(len(flat_env_ids),1),device=self.device).squeeze(1)
            self.commands[flat_env_ids,1]=torch_rand_float(self.command_ranges["flat_lin_vel_y"][0],self.command_ranges["flat_lin_vel_y"][1],(len(flat_env_ids),1),device=self.device).squeeze(1)
            self.commands[flat_env_ids,2]=torch_rand_float(self.command_ranges["flat_ang_vel_yaw"][0],self.command_ranges["flat_ang_vel_yaw"][1],(len(flat_env_ids),1),device=self.device).squeeze(1)
        self.commands[env_ids,:2]*=(torch.norm(self.commands[env_ids,:2],dim=1)>0.2).unsqueeze(1)
        zero_ids=env_ids[self._zero_heading_mask[env_ids]]
        if len(zero_ids)>0:
            self.commands[zero_ids,3]=0
    def _reward_cheat(self):
        if not getattr(self,'_custom_terrain_masks',False):
            return super()._reward_cheat()
        forward=quat_apply(self.base_quat,self.forward_vec)
        heading=torch.atan2(forward[:,1],forward[:,0])
        cheat=(heading>1.0)|(heading<-1.0)
        return torch.where(self._straight_mask,cheat.float(),torch.zeros_like(heading))
    def _reward_feet_edge(self):
        if self.terrain is None:return torch.zeros(self.num_envs,device=self.device)
        if not getattr(self,'_custom_terrain_masks',False):
            return super()._reward_feet_edge()
        feet_pos_xy=((self.rigid_body_states.view(self.num_envs,-1,13)[:,self.feet_indices,:2]+self.terrain.cfg.border_size)/self.cfg.terrain.horizontal_scale).round().long()
        feet_pos_xy[...,0]=torch.clip(feet_pos_xy[...,0],0,self.x_edge_mask.shape[0]-1)
        feet_pos_xy[...,1]=torch.clip(feet_pos_xy[...,1],0,self.x_edge_mask.shape[1]-1)
        feet_at_edge=self.x_edge_mask[feet_pos_xy[...,0],feet_pos_xy[...,1]]
        self.feet_at_edge=self.contact_filt&feet_at_edge
        rew=(self.terrain_levels>3)*torch.sum(self.feet_at_edge,dim=-1)
        return torch.where(self._edge_stumble_mask,rew,torch.zeros_like(rew))
    def _reward_feet_stumble(self):
        if not getattr(self,'_custom_terrain_masks',False):
            return super()._reward_feet_stumble()
        rew=torch.any(torch.norm(self.contact_forces[:,self.feet_indices,:2],dim=2)>4*torch.abs(self.contact_forces[:,self.feet_indices,2]),dim=1)
        rew=(rew*(self.terrain_levels>3)).float()
        return torch.where(self._edge_stumble_mask,rew,torch.zeros_like(rew))
