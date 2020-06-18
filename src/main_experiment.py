import click
from experiment import Experiment
from experimentset import ExperimentSet
import tensorflow as tf
from keras import backend as K

@click.group()
def cli():
	pass


@cli.command('experiment', help='Experiment mode')
@click.option('--file', '-f', required=True, help=u'File that contains the experiments that will be executed.')
@click.option('--gpu', '-g', required=False, default=0, help=u'GPU index')
def experiment(file, gpu):
	experimentSet = ExperimentSet(file)
	experimentSet.run_all(gpu_number=gpu)


if __name__ == '__main__':
	cli()