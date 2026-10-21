// Execute with new Function('tools', source) from functions.exec; tool calls remain explicit.
return (async()=>{
 const repo='/Users/harsh/Documents/HW/em_trait_monitor_public';
 const id='19naffBaAzl3zpYNiXgTYrpS_ArmF4AKIMrjNzqaf5Ws';
 const read=await tools.exec_command({cmd:'/Users/harsh/Documents/HW/.venv/bin/python extensions/harmful_fraction/live_deck_content.py',workdir:repo,max_output_tokens:200});
 if(read.exit_code!==0)throw Error(read.output);
 const raw=await tools.exec_command({cmd:'cat extensions/harmful_fraction/deliverables/live/deck_content.json',workdir:repo,max_output_tokens:12000});
 const c=JSON.parse(raw.output);
 const live=await tools.mcp__codex_apps__google_drive_get_presentation({presentation_id:id});
 if(live.isError)throw Error(JSON.stringify(live));
 const d=live.structuredContent;let requests=[];
 for(const s of d.slides)for(const e of s.pageElements)if(c.text[e.objectId]!==undefined){
   const old=e.shape.text.textElements.map(x=>x.textRun?.content||'').join('').trimEnd();
   if(old!==c.text[e.objectId]) {
    if(old.length) requests.push({deleteText:{objectId:e.objectId,textRange:{type:'FIXED_RANGE',startIndex:0,endIndex:old.length}}});
    requests.push({insertText:{objectId:e.objectId,text:c.text[e.objectId],insertionIndex:0}});
    const sourceStyle=e.shape.text.textElements.find(t=>t.textRun?.content.trim())?.textRun.style||{};
    const style={};for(const key of ['fontFamily','fontSize','foregroundColor','bold','italic'])if(sourceStyle[key]!==undefined)style[key]=sourceStyle[key];
    if(Object.keys(style).length)requests.push({updateTextStyle:{objectId:e.objectId,textRange:{type:'ALL'},style,fields:Object.keys(style).join(',')}});
   }
 }
 const table=d.slides.find(s=>s.objectId==='p2').pageElements.find(e=>e.objectId==='p2_i8').table;
 if(table.rows!==8)throw Error('Unexpected table structure; inspect before editing');
 for(let r=0;r<8;r++)for(let col=0;col<3;col++){
  const old=table.tableRows[r].tableCells[col].text.textElements.map(t=>t.textRun?.content||'').join('').trimEnd();
  if(old!==c.table[r][col]){if(old)requests.push({deleteText:{objectId:'p2_i8',cellLocation:{rowIndex:r,columnIndex:col},textRange:{type:'ALL'}}});requests.push({insertText:{objectId:'p2_i8',cellLocation:{rowIndex:r,columnIndex:col},text:c.table[r][col],insertionIndex:0}});}
  requests.push({updateTextStyle:{objectId:'p2_i8',cellLocation:{rowIndex:r,columnIndex:col},textRange:{type:'ALL'},style:{fontFamily:'Helvetica Neue',fontSize:{magnitude:17,unit:'PT'},bold:r===0,foregroundColor:{opaqueColor:{rgbColor:r===0?{red:1,green:1,blue:1}:{red:.08235,green:.17647,blue:.27059}}}},fields:'fontFamily,fontSize,bold,foregroundColor'}});
 }
 for(const s of d.slides){const notes=s.slideProperties.notesPage;const objectId=notes.notesProperties.speakerNotesObjectId;const el=notes.pageElements?.find(e=>e.objectId===objectId);if(el?.shape?.text?.textElements?.some(t=>t.textRun?.content.trim()))requests.push({deleteText:{objectId,textRange:{type:'ALL'}}});requests.push({insertText:{objectId,text:c.notes,insertionIndex:0}});}
 requests.push({updateTextStyle:{objectId:'p4_i10',textRange:{type:'ALL'},style:{fontFamily:'Helvetica Neue',fontSize:{magnitude:18,unit:'PT'},foregroundColor:{opaqueColor:{rgbColor:{red:.32549,green:.4,blue:.47059}}}},fields:'fontFamily,fontSize,foregroundColor'}});
 const result=await tools.mcp__codex_apps__google_drive_batch_update_presentation({presentation_id:id,requests,write_control:{requiredRevisionId:d.revisionId}});
 if(result.isError)throw Error(JSON.stringify(result));
 const previous=await tools.exec_command({cmd:'cat extensions/harmful_fraction/deliverables/live/published.json',workdir:repo,max_output_tokens:12000});
 const changedCharts=previous.exit_code!==0 || JSON.stringify(JSON.parse(previous.output).distinct_seed_counts)!==JSON.stringify(c.summary.distinct_seed_counts);
 for(const metric of (changedCharts?['pc1','honesty','helpfulness','harmlessness','power_seeking','corrigibility','sycophancy','confidence']:[])){
  const path=repo+'/extensions/harmful_fraction/deliverables/live/'+metric+'.png';
  const r=await tools.mcp__codex_apps__google_drive_batch_update_presentation({presentation_id:id,image_uris:path,requests:[{replaceImage:{imageObjectId:metric==='pc1'?'verified_chart_overall':'small_chart_'+metric,url:path,imageReplaceMethod:'CENTER_INSIDE'}}]});
  if(r.isError)throw Error(JSON.stringify(r));
 }
 return c.summary;
})();
