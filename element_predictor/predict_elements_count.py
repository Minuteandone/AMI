"""Name-only count-aware element prediction.

Usage:
  python predict_elements_count.py Bumbleler
  python predict_elements_count.py "bb$obbinn"
"""
import argparse,json,torch
from count_aware import load,inference

def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser()
    p.add_argument('name',help='Monster name; no element count is required')
    a=p.parse_args()
    print(json.dumps(inference(a.name,load()),indent=2,ensure_ascii=False))
if __name__=='__main__': main()
