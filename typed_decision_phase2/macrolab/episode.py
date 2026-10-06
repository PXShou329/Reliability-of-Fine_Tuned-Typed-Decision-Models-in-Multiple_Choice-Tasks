"""Public episode policy. Programs/FINISH/KEEP are actual model outputs, never hidden-ranked."""
from interpreter import PRIMITIVES
PRIMITIVE_HELP='ABS:absolute; ADD_ONE:add1; DOUBLE:times2; FILTER_EVEN:keep even; FILTER_POSITIVE:keep >0; SORT_ASC:sort ascending; REVERSE:reverse; DEDUP_STABLE:unique first.'
def state_text(demos,program,feedback,library,condition,previous_programs=()):
    import json
    text='Infer a transformation from examples. Up to4 trials of at most4 operations. FINISH submits; KEEP ends with that earlier program. '+PRIMITIVE_HELP
    text+=' Examples:'+json.dumps(demos,separators=(',',':'))
    text+=' Current:'+','.join(program)
    if previous_programs:text+=' Earlier:'+';'.join('candidate'+str(i)+':'+','.join(p) for i,p in enumerate(previous_programs))
    if condition=='M2' and library:text+=' Library:'+';'.join('M'+str(i)+'='+','.join(p) for i,p in enumerate(library))
    if condition in ['M1','M2'] and feedback:
        # Fixed compact observation: most recent support result and every previously passing candidate.
        last=feedback[-1];text+=' Last program:'+','.join(last['program'])+' passed:'+str(last['feedback']['passed'])+'/5'
        c=last['feedback']['counterexample']
        if c:text+=' Counterexample:'+json.dumps(c,separators=(',',':'))
        passed=['candidate'+str(i)+':'+','.join(x['program']) for i,x in enumerate(feedback) if x['feedback']['passed']==5]
        if passed:text+=' Passed:'+';'.join(passed)
    return text
def episode(model,public,condition,verifier,library):
    import time
    start=time.perf_counter()
    traces=[];feedback=[];complete=[];program=[];selected=None;finished=0
    for call in range(20):
        ops=[('append',p) for p in PRIMITIVES if len(program)<4]
        if condition=='M2':ops += [('library',i) for i,p in enumerate(library) if len(program)+len(p)<=4]
        ops += [('finish',None)]
        if condition in ['M1','M2']:ops += [('keep',i) for i,x in enumerate(feedback) if x['feedback']['passed']==5]
        descriptions=[p if op=='append' else ('M'+str(p) if op=='library' else ('FINISH' if op=='finish' else 'KEEP candidate'+str(p))) for op,p in ops]
        action=model.decide(state_text(public['demos'],program,feedback,library,condition,complete),descriptions);op,param=ops[action['selected_index']]
        trace={'call':call+1,'before_program':program.copy(),'action':op,'parameter':param,**action};traces.append(trace)
        if op=='append':program.append(param)
        elif op=='library':program.extend(library[param])
        elif op=='keep':selected=feedback[param]['program'].copy();trace['episode_finished_by_model']=True;break
        else:
            candidate=program.copy();complete.append(candidate);finished+=1;result=verifier.verify(candidate)
            # M0 records permitted evaluator support privately, never adds it to future observation.
            trace['support_result']=result;trace['support_visible']=condition!='M0'
            if condition!='M0':feedback.append({'program':candidate,'feedback':result})
            program=[]
            if finished>=4:break
    if selected is None:selected=complete[-1] if complete else None
    assert len(traces)<=20 and verifier.calls<=4 and (selected is None or len(selected)<=4)
    return {'episode_id':public['episode_id'],'split':public['split'],'condition':condition,'model':model.name,'traces':traces,
            'candidate_programs':complete,'selected_program':selected,'model_calls':len(traces),'verifier_calls':verifier.calls,
            'library_action_n':sum(t['action']=='library' for t in traces),'latency_ms':sum(t['latency_ms'] for t in traces),
            'episode_wall_ms':(time.perf_counter()-start)*1000,'all_actions_actual_model_outputs':True,'hidden_access':False}
