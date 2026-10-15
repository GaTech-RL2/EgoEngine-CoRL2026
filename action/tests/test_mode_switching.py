import ast
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass, field, replace
import io
import json
import math
from pathlib import Path
import tempfile
import types
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import mujoco
import torch

from spider.mode_switching import (planning_windows, select_mode, WorkLedger,
                                  OptimizationClock, FarthestProgress, LegacyTrainingTime)

ROOT = Path(__file__).resolve().parents[1]


def load_runner_functions():
    """Use the actual runner functions with injectable backends, without CUDA imports."""
    tree = ast.parse((ROOT / 'examples/run_mjwp_mixed.py').read_text())
    classes = {'MixedArgs', 'EnvSnapshot', 'WindowSpec', 'WindowResult', 'MixedResult'}
    functions = {'main', '_save_result', '_window_succeeded', '_valid_window_steps', '_finalize_window_result',
                 '_stack_or_empty', '_array_or_empty', '_parse_methods', '_ordered_methods',
                 '_execute_end', '_json_ready', '_run_replay_window', '_run_mpc_window',
                 '_run_rl_window', '_snapshot_env_state', '_restore_env_state', '_slice_ref',
                 '_make_ref_step', '_record_step', '_error_from_reward_info',
                 '_evaluate_rl_window'}
    selected = [n for n in tree.body if (isinstance(n, ast.ClassDef) and n.name in classes)
                or (isinstance(n, ast.FunctionDef) and n.name in functions)]
    ns = dict(np=np, torch=torch, mujoco=mujoco, dataclass=dataclass, field=field, replace=replace,
              Any=Any, Config=SimpleNamespace, Counter=Counter, Path=Path, json=json, math=math,
              io=io, redirect_stdout=redirect_stdout,
              planning_windows=planning_windows, select_mode=select_mode,
              WorkLedger=WorkLedger, OptimizationClock=OptimizationClock,
              FarthestProgress=FarthestProgress, LegacyTrainingTime=LegacyTrainingTime,
              METHOD_TO_ID={'replay': 0, 'mpc': 1, 'rl': 2},
              cprint=lambda *a: None, _synchronize=lambda *a: None)
    ns['adapt_mlp_config']=lambda config:config
    ns.update(get_qpos=lambda *args:torch.zeros((1,1)),
              get_qvel=lambda *args:torch.zeros((1,1)))
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<mixed-runner>', 'exec'), ns)
    return ns


class SwitchingTest(unittest.TestCase):
    def test_control_units_lookahead_and_short_tail(self):
        windows = planning_windows(450, 5)
        self.assertEqual([(w.start, w.execute_end, w.plan_end) for w in windows],
                         [(0,100,200), (100,200,300), (200,300,400), (300,400,450), (400,450,450)])
        self.assertEqual(sum(w.execute_end-w.start for w in windows), 450)

    def test_failed_probes_are_rolled_back_and_only_prefix_is_committed(self):
        state = {'position': 7}
        starts = []
        def reset(): state['position'] = 7
        def solve(mode):
            starts.append(state['position'])
            state['position'] = 999
            return {'mode': mode, 'feasible': mode == 'mpc', 'boundary': 27}
        result, attempts = select_mode(['replay','mpc','rl'], reset, solve,
                                      lambda r:r['feasible'],
                                      lambda r:state.update(position=r['boundary']))
        self.assertEqual(starts, [7,7])
        self.assertEqual(state['position'], 27)
        self.assertEqual(result['mode'], 'mpc')
        self.assertEqual(len(attempts), 2)

    def test_failure_and_exceptions_restore_boundary(self):
        state = [4]
        def reset(): state[0] = 4
        def solve(mode):
            state[0] = 55
            return False
        result, attempts = select_mode(['replay','mpc','rl'], reset, solve, bool,
                                      lambda r: self.fail('must not commit'))
        self.assertIsNone(result); self.assertEqual(len(attempts), 3); self.assertEqual(state[0],4)
        def broken(mode):
            state[0] = 9
            raise RuntimeError('backend failed')
        with self.assertRaises(RuntimeError):
            select_mode(['rl'], reset, broken, bool, lambda r: None)
        self.assertEqual(state[0],4)

    def test_legacy_first_farthest_time_and_hook_exclusion(self):
        now = [0.0]
        clock = OptimizationClock(now=lambda:now[0])
        now[0] = 5
        with clock.exclude(): now[0] = 105
        now[0] = 112
        self.assertEqual(clock.elapsed(),12)
        with clock.exclude():
            now[0]=200
            with clock.exclude():now[0]=300
            now[0]=400
        self.assertEqual(clock.elapsed(),12)
        progress = FarthestProgress()
        progress.observe(20,5); progress.observe(20,12); progress.observe(10,14)
        self.assertEqual((progress.steps,progress.elapsed),(20,5))
        progress.observe(40,15)
        self.assertEqual(progress.elapsed,15)
        legacy = LegacyTrainingTime()
        self.assertEqual(legacy.at_hook(7),0)
        self.assertEqual(legacy.at_hook(19),12)
        self.assertEqual(legacy.elapsed(22),15)
        self.assertEqual(legacy.warmup,7)

    def test_work_counts_worlds_and_keeps_evaluation_separate(self):
        ledger = WorkLedger()
        ledger.record(1,'replay');ledger.record(8,'mpc');ledger.record(32,'rl')
        with ledger.evaluating(): ledger.record(1,'rl')
        ledger.record(32,'rl')
        self.assertEqual(ledger.snapshot()['generation_sim_steps'],73)
        self.assertEqual(ledger.snapshot()['evaluation_sim_steps'],1)
        self.assertEqual(ledger.snapshot()['vector_generation_by_mode']['rl'],2)

    def test_runner_returns_to_replay_and_does_not_duplicate_lookahead(self):
        ns = load_runner_functions()
        state = {'position':0}
        probes = []
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(ctrl_steps=2, ref_steps=2, horizon_steps=0, nu=1,
                                  ref_dt=.02, sim_dt=.01,
                                  output_dir=tmp, device='cpu', data_path='fixture.npz')
            ns['_build_config'] = lambda *a,**k: SimpleNamespace(**vars(cfg))
            ns['load_data'] = lambda *a: (torch.zeros(22,1),)*5
            ns['setup_env'] = lambda *a: state
            ns['_snapshot_env_state'] = lambda env: env['position']
            ns['_restore_env_state'] = lambda cfg,env,snapshot: env.update(position=snapshot)
            ns['make_rollout_fn'] = lambda *a: None
            ns['make_optimize_once_fn'] = lambda *a: None
            ns['make_optimize_fn'] = lambda *a: None
            for name in ('step_env','save_state','load_state','get_reward','get_terminal_reward',
                         'get_terminate','get_trace','save_env_params','load_env_params','copy_sample_state'):
                ns[name] = None
            def trial(config,env,ref,spec):
                probes.append((spec.window_id,spec.method_name,state['position']))
                width = spec.end_step-spec.start_step
                # Replay passes the first chunk but fails its required lookahead.
                if spec.window_id == 0 and spec.method_name == 'replay':
                    width = spec.execute_end_step-spec.start_step
                state['position'] = spec.start_step+width
                for _ in range(width): config.work_ledger.record(1,spec.method_name)
                controls = np.arange(spec.start_step,spec.start_step+width,dtype=np.float32)[:,None]
                return ns['WindowResult'](controls,np.ones(width),np.zeros(width),
                    np.full(width,ns['METHOD_TO_ID'][spec.method_name]),width,False,-1,
                    {'method_name':spec.method_name,'method_id':ns['METHOD_TO_ID'][spec.method_name],
                     'rl_iters_allocated':0,'window_id':spec.window_id},spec.execute_end_step)
            ns['_run_replay_window'] = trial
            ns['_run_mpc_window'] = lambda cfg,env,opt,ref,spec: trial(cfg,env,ref,spec)
            args = ns['MixedArgs'](methods='replay,mpc',device='cpu',window_size=4,
                                   output_name='test.npz')
            with redirect_stdout(io.StringIO()): ns['main'](args)
            with np.load(Path(tmp)/'test.npz') as result:
                np.testing.assert_array_equal(result['ctrl'][:,0],np.arange(20))
                self.assertEqual(result['window_method_id'].tolist(),[1,0,0])
                self.assertEqual(result['window_end'].tolist(),[8,16,20])
                self.assertEqual(result['window_plan_end'].tolist(),[16,20,20])
                attempts=json.loads(result['attempts_json'].item())
                self.assertFalse(attempts[0]['feasible'])
                stats=json.loads(result['statistics_json'].item())
                self.assertEqual(stats['generation_sim_steps'],40)
                self.assertEqual(stats['steps_actual'],10)
                self.assertEqual(stats['cost'],4)
            self.assertEqual(probes,[(0,'replay',0),(0,'mpc',0),(1,'replay',8),(2,'replay',16)])
            self.assertEqual(state['position'],20)

    def test_nonfinite_plan_is_not_feasible(self):
        ns=load_runner_functions();spec=ns['WindowSpec'](0,0,4,'replay')
        result=ns['WindowResult'](np.ones((4,1)),np.ones(4),np.zeros(4),np.zeros(4),4,False,-1)
        self.assertTrue(ns['_window_succeeded'](result,spec))
        result.ctrl[0,0]=np.nan
        self.assertFalse(ns['_window_succeeded'](result,spec))
        result.ctrl[0,0]=1
        result.qpos_states=[torch.ones(1) for _ in range(4)]
        result.qvel_states=[torch.ones(1) for _ in range(4)]
        result.qpos_states[2][0]=float('nan')
        self.assertEqual(ns['_valid_window_steps'](result),2)
        self.assertFalse(ns['_window_succeeded'](result,spec))

    def test_window_budget_and_accepted_checkpoint_survive_a_replay_segment(self):
        ns = load_runner_functions()
        state, calls = {'position':0}, []
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(ctrl_steps=2, ref_steps=2, horizon_steps=0, nu=1,
                                  ref_dt=.02, sim_dt=.01, output_dir=tmp, device='cpu', data_path='fixture.npz')
            ns['_build_config'] = lambda *a,**k: SimpleNamespace(**vars(cfg))
            ns['load_data'] = lambda *a: (torch.zeros(22,1),)*5
            ns['setup_env'] = lambda *a: state
            ns['_snapshot_env_state'] = lambda env: env['position']
            ns['_restore_env_state'] = lambda cfg,env,snapshot: env.update(position=snapshot)

            def trial(config, env, ref, spec, previous_checkpoint=None):
                width = spec.end_step - spec.start_step
                failed = spec.method_name == 'replay' and spec.window_id != 1
                used = min(2, spec.rl_iters_allocated) if spec.method_name == 'rl' else 0
                summary = dict(method_name=spec.method_name, method_id=ns['METHOD_TO_ID'][spec.method_name],
                               window_id=spec.window_id, rl_iterations_used=used, rl_iters_allocated=spec.rl_iters_allocated,
                               continuation_checkpoint=f'checkpoint_{spec.window_id}',
                               warm_started=previous_checkpoint is not None)
                if spec.method_name == 'rl':
                    calls.append((spec.window_id, spec.rl_iters_allocated, previous_checkpoint))
                return ns['WindowResult'](np.ones((width,1)),np.ones(width),np.zeros(width),
                                          np.full(width,ns['METHOD_TO_ID'][spec.method_name]),
                                          width,failed,0 if failed else -1,summary,spec.execute_end_step)

            ns['_run_replay_window'] = trial
            ns['_run_rl_window'] = lambda args, cfg, env, rl_cfg, ref, spec, **kw: trial(cfg,env,ref,spec,**kw)
            for continue_policy, methods in ((True,'replay,rl'),(False,'replay,rl'),(True,'rl')):
                calls.clear()
                state['position'] = 0
                args = ns['MixedArgs'](methods=methods,device='cpu',window_size=4,
                                      rl_total_learning_iterations=5,rl_window_learning_iterations=3,
                                      rl_continue_from_previous=continue_policy,output_name='budget.npz')
                with redirect_stdout(io.StringIO()): ns['main'](args)
                expected = ([(0,3,None),(1,3,'checkpoint_0'),(2,1,'checkpoint_1')] if methods == 'rl'
                            else [(0,3,None),(2,3,'checkpoint_0' if continue_policy else None)])
                self.assertEqual(calls,expected)
                with np.load(Path(tmp)/'budget.npz') as artifact:
                    stats=json.loads(artifact['statistics_json'].item())
                    self.assertEqual(stats['rl_iterations_used'],5 if methods == 'rl' else 4)
                    self.assertEqual(stats['rl_warm_started_windows'],2 if methods == 'rl' else 1 if continue_policy else 0)

    def _exercise_rl_hook(self, last_step_failure=False, warm_start=False):
        ns = load_runner_functions()
        now, state, evaluations = [0.0], [0], []
        ledger = WorkLedger()
        policy = SimpleNamespace(training=True)
        policy.eval = lambda: setattr(policy, 'training', False)
        policy.train = lambda value: setattr(policy, 'training', value)
        policy.state_dict = lambda: {}
        load_calls = []
        class FakeEnv:
            def __init__(self, cfg, data, env_cfg, **kwargs):
                self.num_obs, self.num_actions = 1, 1
            def close(self): pass
        class FakeRunner:
            def __init__(self, env, *args, **kwargs):
                self.alg = SimpleNamespace(get_policy=lambda:policy)
                self.current_learning_iteration = 0
                self.logger = SimpleNamespace(log=lambda **kwargs:None, writer=None)
            def save(self, path):
                now[0] += 50  # native checkpoint IO must remain outside the training clock
                Path(path).write_bytes(b'accepted runner state')
            def load(self, path, load_cfg, map_location):
                self.assert_checkpoint = Path(path).read_bytes()
                load_calls.append((load_cfg, map_location))
                self.current_learning_iteration = 999
                now[0] += 90  # loading is setup, before the optimization clock starts
            def learn(self, num_learning_iterations, **kwargs):
                if self.current_learning_iteration != 0:
                    raise AssertionError('A new window inherited the old iteration counter')
                for iteration in range(num_learning_iterations):
                    now[0] += 7 if iteration == 0 else 5
                    ledger.record(32,'rl')
                    self.current_learning_iteration = iteration
                    self.logger.log(it=iteration)
        runners = types.ModuleType('rsl_rl.runners')
        runners.OnPolicyRunner = FakeRunner
        env_module = types.ModuleType('spider.rl.mjwp_rsl_env')
        env_module.MJWPRslResidualEnv = FakeEnv
        env_module.MJWPRLEnvCfg = lambda **kwargs: SimpleNamespace(**kwargs)
        def evaluate(cfg,env,data,spec,train_env,policy,snapshot):
            self.assertFalse(policy.training)
            self.assertEqual(state[0],snapshot)
            now[0] += 100  # expensive evaluation must not enter optimization time
            ledger.record(1,'rl')
            count = len(evaluations)
            steps = 4 if last_step_failure else 2 if count < 2 else 4
            terminated = last_step_failure and count == 0
            reward = 100.0 if terminated else float(count)
            state[0] = 999
            evaluations.append(count)
            return ns['WindowResult'](np.ones((steps,1)),np.ones(steps),np.zeros(steps),
                    np.full(steps,2),steps,terminated,3 if terminated else -1,{'total_reward':reward},2)
        ns.update(_snapshot_env_state=lambda env:state[0],
                  _restore_env_state=lambda cfg,env,snapshot:state.__setitem__(0,snapshot),
                  _evaluate_rl_window=evaluate, _make_rsl_train_cfg=lambda **kwargs:{},
                  OptimizationClock=lambda *args:OptimizationClock(now=lambda:now[0]))
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules',
                {'rsl_rl.runners':runners,'spider.rl.mjwp_rsl_env':env_module}):
            cfg=SimpleNamespace(work_ledger=ledger,ctrl_steps=1,output_dir=tmp,device='cpu')
            args=ns['MixedArgs'](device='cpu')
            spec=ns['WindowSpec'](0,0,4,'rl',rl_iters_allocated=10,execute_end_step=2)
            checkpoint = Path(tmp) / 'previous.pt'
            checkpoint.write_bytes(b'previous accepted runner state')
            result=ns['_run_rl_window'](args,cfg,None,cfg,None,spec,
                                       previous_checkpoint=str(checkpoint) if warm_start else None)
            self.assertTrue(Path(result.summary['continuation_checkpoint']).is_file())
            if warm_start:
                self.assertEqual(load_calls, [({'actor':True,'critic':True,'optimizer':True,
                                               'iteration':False,'rnd':True}, 'cpu')])
            else:
                self.assertEqual(load_calls, [])
        self.assertTrue(policy.training)
        self.assertEqual(state[0],0)
        return result,ledger

    def test_rl_hook_preserves_warmup_farthest_timing_and_stops_early(self):
        result,ledger=self._exercise_rl_hook()
        self.assertEqual(result.summary['rl_iterations_used'],3)
        self.assertEqual(result.summary['optimization_elapsed'],17)
        self.assertEqual(result.summary['time_to_farthest'],10)
        self.assertEqual(result.summary['warmup_elapsed'],7)
        self.assertEqual(result.summary['hook_excluded_time'],350)
        self.assertEqual(ledger.snapshot()['generation_sim_steps'],96)
        self.assertEqual(ledger.snapshot()['evaluation_sim_steps'],3)

    def test_continuation_loads_full_runner_without_charging_io_or_inheriting_iteration(self):
        result, ledger = self._exercise_rl_hook(warm_start=True)
        self.assertTrue(result.summary['warm_started'])
        self.assertEqual(result.summary['optimization_elapsed'],17)
        self.assertEqual(result.summary['time_to_farthest'],10)
        self.assertEqual(result.summary['rl_iterations_used'],3)

    def test_late_success_replaces_a_higher_reward_last_step_failure(self):
        result,ledger=self._exercise_rl_hook(last_step_failure=True)
        self.assertFalse(result.terminated)
        self.assertEqual(result.summary['rl_iterations_used'],2)
        self.assertEqual(result.summary['time_to_farthest'],5)
        self.assertEqual(result.summary['benchmark_elapsed'],5)

    def test_failed_diagnostics_are_strict_json(self):
        ns = load_runner_functions()
        value = ns['_json_ready']({'feasible':np.bool_(False),'mean_error':float('nan'),
                                  'steps':np.int64(3),'speed':math.inf})
        encoded=json.dumps(value,allow_nan=False)
        self.assertEqual(json.loads(encoded),{'feasible':False,'mean_error':None,'steps':3,'speed':None})

    def test_task_metadata_recomputes_interpolation_ratio(self):
        from spider.config import Config, process_config
        from spider.io import load_data
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'processed/aria/xhand/right/fixture'
            (root/'0').mkdir(parents=True)
            (root/'scene.xml').write_text('<mujoco><worldbody><body><joint name="motor"/><geom type="sphere" size=".1"/></body><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody><actuator><motor joint="motor"/></actuator></mujoco>')
            (root/'task_info.json').write_text('{"ref_dt":0.05}')
            cfg=process_config(Config(dataset_dir=tmp,dataset_name='aria',robot_type='xhand',
                                      embodiment_type='right',task='fixture',simulator='mjwp',
                                      device='cpu',horizon=.1,knot_dt=.05,ctrl_dt=.05))
            self.assertEqual(cfg.ref_steps,5)
            np.savez(root/'0/trajectory_kinematic.npz',qpos=np.zeros((3,8)),
                     qvel=np.zeros((3,7)),ctrl=np.zeros((3,1)),
                     contact=np.zeros((3,10)),contact_pos=np.zeros((3,10,3)))
            refs=load_data(cfg,cfg.data_path)
            self.assertEqual(len(refs[0])-cfg.horizon_steps-cfg.ctrl_steps,15)

    def test_failed_export_is_not_reported_as_a_complete_trajectory(self):
        from action.paper_results_analysis.analyze_mixed_results import load_mixed_result
        ns=load_runner_functions()
        with tempfile.TemporaryDirectory() as tmp:
            result=ns['MixedResult'](np.empty((0,1)),np.empty(0),np.empty(0),
                                     np.empty(0,dtype=np.int64),[],False,
                                     [{'feasible':False,'mean_error':math.nan}],{'success':False})
            path=Path(tmp)/'failed.npz'
            ns['_save_result'](result,path)
            loaded=load_mixed_result(path)
            self.assertTrue(loaded.terminated_early)
            self.assertEqual(loaded.num_windows,0)
            self.assertEqual(loaded.num_steps,0)

    def test_snapshot_restores_runtime_fields_and_broadcasts_boundary(self):
        import mujoco
        ns=load_runner_functions()
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="object"><freejoint/><geom type="sphere" size=".1"/></body></worldbody><equality><weld body1="object" active="false"/></equality></mujoco>')
        mj_data=mujoco.MjData(model)
        fields={name:torch.as_tensor(getattr(mj_data,name).copy()).unsqueeze(0).repeat(
                    (3,)+(1,)*getattr(mj_data,name).ndim)
                for name in ('qpos','qvel','ctrl','act','qacc_warmstart','mocap_pos',
                             'mocap_quat','eq_active','qfrc_applied','xfrc_applied')}
        fields['time']=torch.ones(3)
        fields['qacc_warmstart'].fill_(7)
        fields['qfrc_applied'].fill_(2)
        fields['eq_active'].fill_(True)
        env=SimpleNamespace(model_cpu=model,data_wp=SimpleNamespace(**fields))
        def sync(cfg,env,data):
            # Generic sync intentionally omits equality and extra runtime fields.
            for name in ('qpos','qvel','ctrl'):
                getattr(env.data_wp,name).copy_(torch.as_tensor(getattr(data,name)).expand_as(getattr(env.data_wp,name)))
        ns.update(mujoco=mujoco,wp=SimpleNamespace(to_torch=lambda value:value),
                  get_qpos=lambda cfg,env:env.data_wp.qpos,
                  get_qvel=lambda cfg,env:env.data_wp.qvel,sync_env_mujoco=sync)
        saved=ns['_snapshot_env_state'](env)
        for name in ('qacc_warmstart','qfrc_applied','eq_active'):
            getattr(env.data_wp,name).zero_()
        env.data_wp.qpos[:,0]=5
        ns['_restore_env_state'](SimpleNamespace(),env,saved)
        self.assertEqual(env.data_wp.qacc_warmstart[2,0],7)
        self.assertEqual(env.data_wp.qfrc_applied[2,0],2)
        self.assertTrue(env.data_wp.eq_active[2,0])
        self.assertEqual(env.data_wp.qpos[2,0],0)

    def test_real_replay_and_mpc_functions_capture_commit_boundary(self):
        ns=load_runner_functions();ledger=WorkLedger()
        cfg=SimpleNamespace(nu=1,ctrl_steps=2,horizon_steps=4,work_ledger=ledger)
        env=SimpleNamespace(position=0,num_worlds=1)
        def step(config,env,control): env.position+=1;ledger.record(1,'test')
        ns.update(step_env=step,reset_inactive_side_to_ref=lambda *a:None,
                  get_reward=lambda *a:(torch.ones(1),{'total_dist':torch.zeros(1)}),
                  get_terminate=lambda *a:torch.tensor([False]),
                  _snapshot_env_state=lambda e:e.position)
        ns.update(get_qpos=lambda c,e:torch.tensor([[float(e.position)]]),
                  get_qvel=lambda c,e:torch.zeros((1,1)))
        spec=ns['WindowSpec'](0,0,8,'replay',execute_end_step=4)
        data=(torch.zeros(12,1),)*5
        replay=ns['_run_replay_window'](cfg,env,data,spec)
        self.assertEqual((replay.commit_state,env.position,replay.steps_executed),(4,8,8))
        self.assertEqual(torch.stack(replay.qpos_states).flatten().tolist(),list(range(1,9)))
        env.position=0;spec.method_name='mpc'
        result=ns['_run_mpc_window'](cfg,env,lambda c,e,u,r:(u,{}),data,spec)
        self.assertEqual((result.commit_state,env.position,result.steps_executed),(4,8,8))

    def test_rl_evaluation_validates_post_step_reference_and_shared_controls(self):
        ns = load_runner_functions()
        env = SimpleNamespace(position=0)
        cfg = SimpleNamespace(ctrl_steps=2, nu=1, device='cpu')
        refs = (torch.arange(7, dtype=torch.float32)[:,None],) * 5
        control_phases = []
        def controls(actions, phase):
            control_phases.append(phase)
            return torch.zeros(1,1), None, None
        train = SimpleNamespace(env_cfg=SimpleNamespace(action_clip=1),
                                residual_actions_to_ctrl=controls)
        ns.update(_restore_env_state=lambda *a: setattr(env,'position',0),
                  _build_eval_obs=lambda *a: torch.zeros(1,1),
                  step_env=lambda *a: setattr(env,'position',env.position+1),
                  reset_inactive_side_to_ref=lambda *a:None,
                  get_reward=lambda *a:(torch.ones(1),{'total_dist':torch.zeros(1)}),
                  get_terminate=lambda c,e,r: torch.tensor([e.position != int(r[0][0])]),
                  get_qpos=lambda c,e:torch.tensor([float(e.position)]).reshape(1,1),
                  get_qvel=lambda *a:torch.zeros(1,1),
                  _snapshot_env_state=lambda e:e.position)
        spec = ns['WindowSpec'](0,0,4,'rl',execute_end_step=2)
        result = ns['_evaluate_rl_window'](cfg,env,refs,spec,train,
                                          lambda obs:torch.zeros(1,1),0)
        self.assertFalse(result.terminated)
        self.assertEqual(result.steps_executed,4)
        self.assertEqual(result.commit_state,2)
        self.assertEqual(control_phases,[0,2])


if __name__=='__main__': unittest.main()
