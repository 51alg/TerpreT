#!/usr/bin/env python2

'''
Usage:
    benchmark.py [options] RUNFILE...

Options:
    -h --help           Show this screen.
    --profile PROFILE   IPCluster profile. [default: default]
    --dry-run           Print jobs without running them.
    --store-result      Store training result next to log file.
    --out-dir DIRECTORY Output log/run files in DIRECTORY [default: .]
'''

import copy
from glob import glob
import json
import os
import subprocess
import tempfile
from docopt import docopt
import pprint
import ipyparallel as ipp
from ipyparallel import require
import itertools
import datetime
import sys

BASE_PATH = os.path.dirname(__file__)
if sys.platform.startswith('linux'):
    TIMEOUT_PATH = '/mnt/nfs/users/mabrocks/bin/timeout_linux.native'
elif sys.platform == 'darwin':
    TIMEOUT_PATH = 'timeout_osx'
else:
    raise RuntimeError('Unsupported platform: ' + sys.platform)
NICENESS = 15

# Megabytes
INITIAL_MEM_LIMIT = 5000
MAX_MEM = 50000

STARTED_LOG = 'jobs-started.log'
DONE_LOG = 'jobs-done.log'
FAILED_LOG = 'jobs-failed.log'


def run_job_to_cmd(job):
    timeout_cmd = [TIMEOUT_PATH, '-m', str(job['max_mem']), '--']
    nice_cmd = ['nice', '-n', str(NICENESS)]
    train_cmd = ['python', '-u', job['trainer']]
    if 'result_path' in job:
        train_cmd.extend(['--store-data', job['result_path']])
    train_cmd.extend(job['params'])
    train_cmd.append(job['model_path'])
    train_cmd.append(job['data_path'])
    hypers_path = job.get('hypers_path', "")
    if hypers_path != "":
        train_cmd.append(hypers_path)

    cmd = timeout_cmd + nice_cmd + train_cmd
    return cmd


class BenchmarkRunner:
    def __init__(self, out_dir=None, profile='default'):
        self.__out_dir = out_dir
        self.__client = ipp.Client(profile=profile)
        self.__client[:]['run_job_to_cmd'] = run_job_to_cmd

    def log(self, name, job):
        if self.__out_dir is not None:
            name = os.path.join(self.__out_dir, name)
            with open(name, 'a') as f:
                json.dump(job, f)
                f.write('\n')

    def log_started(self, job):
        self.log(STARTED_LOG, job)

    def log_done(self, job):
        self.log(DONE_LOG, job)

    def log_failed(self, job):
        self.log(FAILED_LOG, job)

    def get_engines_by_hostname(self):
        def get_hostname():
            import socket
            return socket.gethostname()
        engine_to_host = self.__client[:].apply_async(get_hostname).get_dict()
        host_to_engines = {}
        for engine in engine_to_host:
            host = engine_to_host[engine]
            host_to_engines[host] = host_to_engines.setdefault(host, []) + [engine]
        return host_to_engines

    def select_engines(self, hosts, num_engines):
        hosts = copy.deepcopy(hosts)
        selected = []
        host_names = itertools.cycle(hosts.keys())
        while len(selected) < num_engines:
            host_name = next(host_names)
            if len(hosts[host_name]) > 0:
                selected.append(hosts[host_name][0])
                hosts[host_name] = hosts[host_name][1:]
        return selected

    def run_memory_intensive(self, jobs, func):
        cap = INITIAL_MEM_LIMIT
        host_to_engines = self.get_engines_by_hostname()

        num_engines = 0
        for host in host_to_engines:
            num_engines += len(host_to_engines[host])

        for job in jobs:
            job['max_mem'] = cap

        while len(jobs) > 0:
            print 'Running {} jobs on {} engines with {} Gb per job.'.\
                format(len(jobs), num_engines, cap / 1000.0)

            engines = self.select_engines(host_to_engines, num_engines)
            lview = self.__client.load_balanced_view(targets=engines)

            ret = []
            for job in jobs:
                self.log_started(job)
                ret.append((job, lview.apply(func, job)))
            self.__client.wait_interactive()

            failed_jobs = []
            for job, job_ret in ret:
                if job_ret.successful():
                    self.log_done(job)
                else:
                    failed_jobs.append(job)

            cap *= 2
            num_engines /= 2

            if cap >= MAX_MEM or num_engines <= 0:
                for job in failed_jobs:
                    self.log_failed(' '.join(['"' + a.replace('"', '\\"') + '"' if ' ' in a or '"' in a else a for a in run_job_to_cmd(job)]))
            else:
                for job in failed_jobs:
                    job['max_mem'] = cap

            jobs = failed_jobs

    @require('os', 'subprocess', 'json', NICENESS=NICENESS, TIMEOUT_PATH=TIMEOUT_PATH)
    def run_train_job(job):
        os.chdir(job['cwd'])
        with open(job['log_path'], 'w') as log:
            cmd = run_job_to_cmd(job)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            p = subprocess.Popen(cmd, stdout=log, stderr=log, env=env)
            p.wait()
            if p.returncode != 0:
                raise RuntimeError('Job process exited with return code {}.'.format(p.returncode))

def write_git_version():
    with open('rev', 'w') as f:
        subprocess.call(['git', 'rev-parse', 'HEAD'], stdout=f)

def main(args):
    # Docopt default parameter values seem to be broken.
    out_dir = args['--out-dir'] or '.'
    store_result = args.get('--store-result', False) or False

    train_jobs = []

    for runfile in args['RUNFILE']:
        with open(runfile, 'r') as f:
            runfile_data = json.load(f)

        trainer = runfile_data['trainer']
        num_restarts = runfile_data['num_restarts']
        global_params = runfile_data.get('global_params', [])
        for run in runfile_data['runs']:
            #compiled_model, data, hypers, params
            model_path = run['compiled_model']
            data_path = run['data']
            run_out_dir = run.get('out_dir', ".")
            hypers_path = run.get('hypers', "")
            params = global_params + run.get('params', [])

            for model_path in glob(model_path):
                for data_path in glob(data_path):
                    model_name = os.path.splitext(os.path.basename(model_path))[0]
                    data_name = os.path.splitext(os.path.basename(data_path))[0]

                    for i in range(num_restarts):
                        base_dir = os.path.join(out_dir, run_out_dir)
                        try:
                            os.makedirs(base_dir)
                        except:
                            pass
                        base_out_path = os.path.join(base_dir, '{}_{}-run_{}'.format(model_name, data_name, i))
                        result_path = base_out_path + ".hdf5"
                        log_path = base_out_path + ".log"
                        train_job = {
                            'cwd': os.getcwd(),
                            'trainer': trainer,
                            'model_path': model_path,
                            'data_path': data_path,
                            'hypers_path': hypers_path,
                            'log_path': log_path,
                            'params' : params,
                            'max_mem' : INITIAL_MEM_LIMIT,
                        }
                        if store_result:
                            train_job['result_path'] = result_path
                        train_jobs.append(train_job)

    if args['--dry-run']:
        pprint.pprint(train_jobs)
        return

    #write_git_version()

    runner = BenchmarkRunner(out_dir, args.get('--profile', 'default'))

    print 'Running...'
    #For debugging:
    #runner.run_train_job(train_jobs[0])
    runner.run_memory_intensive(train_jobs, runner.run_train_job)

if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
