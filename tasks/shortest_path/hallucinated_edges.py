"""Features of the first nonexistent (non-reversed) edge in right-endpoint answers vs random non-edges in the same graph.
Hard-coded to the 0.6B lr3e-6 step-234 eval; edit R to reuse. See research log 2026-11-22_shortest-path-task.md."""
import sys, random, collections
sys.path.insert(0,'tasks/shortest_path')
from analyze_errors import load_results
from reward import extract_answer, parse_path
from graphs import bfs
R='results/Qwen3-0.6B-Base/spath-simple-grpo-lr3e-6-kl3e-3-seed1/global_step_234/max512/'
rng=random.Random(0)
def feats(a,b,src,dst,s,t,dist_from_a,radj_to_t,dist_from_s):
    E=set(zip(src,dst)); nodes=set(src)|set(dst)
    f={}
    f['reverse b->a exists']=(b,a) in E
    da=dist_from_a.get(b)
    f['a reaches b in 2 hops']=da==2
    f['a reaches b in 3+ hops']=da is not None and da>=3
    f['b unreachable from a']=da is None
    f['b == target']=b==t
    f['a == source']=a==s
    # listing adjacency: "x -> a, b -> y" or "a -> x, y -> b"? check consecutive edges
    L=list(zip(src,dst)); adj=False
    for (x1,y1),(x2,y2) in zip(L,L[1:]):
        if (y1==a and x2==b) or (x1==a and y2==b) or (y1==b and x2==a): adj=True
    f['a,b in consecutive listed edges']=adj
    f['a,b share an in-neighbour']=any((x,a) in E and (x,b) in E for x in nodes)
    f['a,b share an out-neighbour']=any((a,y) in E and (b,y) in E for y in nodes)
    f['b one step closer to t than a should be']=None
    f['labels alphabetically adjacent']=abs(ord(a)-ord(b))==1
    return f
for ds in ['spath-simple','spath5-simple','spath6-simple']:
    H=collections.defaultdict(collections.Counter); B=collections.defaultdict(collections.Counter); nH=collections.Counter(); nB=collections.Counter()
    pos=collections.defaultdict(collections.Counter)
    for _,txt,gt in load_results([R+ds+'_temp0.6_n128_max512.json']):
        p=parse_path(extract_answer(txt))
        if p is None: continue
        src=[str(x) for x in gt['edges_src']]; dst=[str(x) for x in gt['edges_dst']]
        E=set(zip(src,dst)); s,t,d=str(gt['source']),str(gt['target']),int(gt['distance'])
        if p[0]!=s or p[-1]!=t: continue
        bad=[(i,a,b) for i,(a,b) in enumerate(zip(p,p[1:])) if (a,b) not in E and (b,a) not in E]
        if not bad: continue
        adj=collections.defaultdict(list)
        for x,y in E: adj[x].append(y)
        i,a,b=bad[0]
        nodes=sorted(set(src)|set(dst))
        H[d].update(k for k,v in feats(a,b,src,dst,s,t,bfs(adj,a)[0],None,None).items() if v); nH[d]+=1
        hops=len(p)-1
        pos[d]['last hop (into target)' if i==hops-1 else 'first hop' if i==0 else 'middle']+=1
        # baseline: random non-edge, non-reverse pair in same graph
        while True:
            a2,b2=rng.sample(nodes,2)
            if (a2,b2) not in E and (b2,a2) not in E: break
        B[d].update(k for k,v in feats(a2,b2,src,dst,s,t,bfs(adj,a2)[0],None,None).items() if v); nB[d]+=1
    for d in sorted(nH):
        print(f"\n== d={d}: {nH[d]} answers with right endpoints whose first invalid edge is not a reversal")
        keys=['a reaches b in 2 hops','a reaches b in 3+ hops','b unreachable from a','b == target','a == source','a,b share an in-neighbour','a,b share an out-neighbour','a,b in consecutive listed edges','labels alphabetically adjacent']
        for k in keys: print(f"   {k:34s} halluc {H[d][k]/nH[d]:6.1%}   random non-edge {B[d][k]/nB[d]:6.1%}")
        print('   position:', {k:f"{v/nH[d]:.0%}" for k,v in pos[d].items()})
