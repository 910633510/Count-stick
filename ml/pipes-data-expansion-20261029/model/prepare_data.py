"""Prepare only the exact newly frozen manifest named by a root freeze receipt."""
from pathlib import Path
import argparse,json
import pipe_data_expansion as p
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args();print(json.dumps(p.prepare(args.receipt),indent=2))
