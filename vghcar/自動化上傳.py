#!/usr/bin/env python
from argparse import ArgumentParser
from datetime import datetime
from sys import argv
from os.path import abspath, dirname
from os import system
from datetime import timedelta
from stUtil import rndrCode

cwd, rscript=dirname(abspath(__file__)), 'demoTrans.r'

def download(args):
	date, dateRange=datetime.strptime(args.date, '%Y-%m-%d'), args.dateRange	#2018-04-29
	for iter in range(dateRange):
		#date=date.split()[0]
		Date=datetime.strftime(date, '%Y-%m-%d')
		#if date<10:date=''.join(['0', str(date)])
		fname='AllCancer%s.csv'%Date
		cmd='scp bhlee@10.97.235.41:/tmp/%s ../VGHCAR_DATASET/'%fname
		#cmd='scp -p bhlee bhlee@10.97.235.41:/tmp/AllCancer%s.csv ../VGHCAR_DATASET/'%Date
		date+=timedelta(days=1)
		rndrCode(cmd)
		system(cmd)
		cmd='Rscript %s ../VGHCAR_DATASET/%s'%(rscript, fname)
		print(cmd)
		system(cmd)

def upload(args):
	for file in glob('20'):
		cmd='scp -P 4822 -p bhlee %s/VGHCAR_DATASET/%s.csv bhlee@10.97.249.120:/docker/vghcar/dataset'%(cwd, file)
		print(cmd)

if __name__=='__main__':
    parser = ArgumentParser(description='calculate stock to the total of SKY')
    parser.add_argument('--download', '-D', action='store_true', help='download')
    parser.add_argument('--upload', '-U', action='store_true', help='upload')
    parser.add_argument('--date', '-t', type=str, help='initDate')
    parser.add_argument('--dateRange', '-r', type=int, default=5, help='initDate')
    args = parser.parse_args()
    if args.download: download(args)
    elif args.upload: upload(args)
