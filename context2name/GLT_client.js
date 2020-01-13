var scoper = require(__dirname + '/scoper.js');
var pq = require('js-priority-queue');

var fs = require('fs');
var esprima = require('esprima');
var estraverse = require('estraverse');
var getParent = require('estree-parent')
var HashMap = require('hashmap');

var syncrequest = require('sync-request');
var escodegen = require('escodegen');

var ArgumentParser = require('argparse').ArgumentParser;
var path = require('path');

var HOP = function (obj, prop) {
  return Object.prototype.hasOwnProperty.call(obj, prop);
};

let ascii_number = 33;
let nodeNameMap = new Object(null);
let globalSeqHashMapWrapper = new Object(null);

const DIVIDER = "区"


//
// Queue (FIFO)
//

function Queue() {
	this.__a = new Array();
}

Queue.prototype.enqueue = function(o) {
	this.__a.push(o);
}

Queue.prototype.dequeue = function() {
	if( this.__a.length > 0 ) {
		return this.__a.shift();
	}
	return null;
}

Queue.prototype.size = function() {
	return this.__a.length;
}

Queue.prototype.toString = function() {
	return '[' + this.__a.join(',') + ']';
}

// declaration of number generator

function* numbers(){
  let i=0;
  while(true){
    yield i;
    i += 1;
  }
}

function getStringFromEdge(res){
  if(res["type"] == "var-var"){
    return res["xScopeId"] + DIVIDER + res["xName"] + DIVIDER + res["yName"] + DIVIDER + res["yScopeId"];
  }
  return res["xScopeId"] + DIVIDER + res["xName"] + DIVIDER + res["yName"];
}

function makeChildParentRelation(ast){
  estraverse.traverse(ast,{
    enter : function(node, parent){
      // node does not have children property
      if(parent){
        if(!("children" in parent)){
          parent["children"] = [];
        }
        parent["children"].push(node);
      }

      if(!("parent" in node)){
        node["parent"] = [];
      }
      node["parent"] = parent;
    }
  })
}


function getNodeTokenOfSequence(node, nodeNameMap){
  let nodetype;
  // if nodenamemap does not contain node.type, add to dic.
  if(node.type == "BinaryExpression"){
    nodetype = node.operator;
  }
  else{
    nodetype = node.type;
  }

  if(!(node.type in nodeNameMap)){
    nodeNameMap[nodetype] = String.fromCharCode(ascii_number);
    ascii_number += 1;
  }
  let add_token = nodeNameMap[node.type];
  return add_token;
}

function getJsonElementFromTwoNode(node1, node2, seq, childNodeType=null){
  // node1 must be id.

  let res;

  if(childNodeType=="element"){
    // node2 is element.
    let name;
    if(node2.type === "Literal"){
      name = node2["raw"];
    }
    else if(node2.type === "ArrayExpression"){
      name = "Array";
    }
    else{
      name = node2["name"];
    }
    res = {"type":"var-lit", "xName":node1.name, "xScopeId":node1.scopeid, "yName":name, "sequence": seq };
  }
  else{
    // node2 is id.
    res = {"type":"var-var", "xName":node1.name, "xScopeId":node1.scopeid, "yName":node2.name, "yScopeId":node2.scopeid, "sequence": seq };
  }
  return res;
}

function newExtractNodeSequences(ast, tokens, rangeToTokensIndexMap, number, scopeParentMap){
  function getNextIteration(node, checkInvoker=null){
    let childrens;
    if (node.children){
      childrens = node.children;
    }
    else{
      // this node is leaf, break.
      childrens = [];
      // return [];
    }

    if(node.parent){
      childrens.push(node.parent);
    }
    childrens = childrens.filter(node => node.range !== checkInvoker.range);
    return childrens;
  }

  function main_process(node, main_invoker, seqMap, seqHashSet, sequence, duplicateCheck, MAX_DISTANCE=5){
    // if sequence length is greater than MAX_DISTANCE, return.
    if(sequence.length >= MAX_DISTANCE) return;

    let childrens = getNextIteration(node, checkInvoker=node);
    childrens.forEach( function(childNode){
      let childNodeType = childNode.isInfer;
      if(childNode.type == "BlockStatement"){
        // when child is not element or id, then check child's child
        main_process(childNode, main_invoker, seqMap, seqHashSet, sequence, duplicateCheck, MAX_DISTANCE=MAX_DISTANCE);
        return;
      }

      let newToken = getNodeTokenOfSequence(childNode, nodeNameMap);
      // this part may be too naive.
      let newSeq = sequence + newToken;

      // check duplicate.
      let range1 = childNode.range[0].toString();
      let range2 = childNode.range[1].toString();
      let rangeToken = range1 + DIVIDER + range2;
      if(duplicateCheck.has(rangeToken)){
        return;
      }
      else{
        duplicateCheck.add(rangeToken)
      }

      if(typeof childNodeType === "undefined"){
        // when child is not element or id, then check child's child
        main_process(childNode, main_invoker, seqMap, seqHashSet, newSeq, duplicateCheck, MAX_DISTANCE=MAX_DISTANCE);
        return;
      }

      // child is element or id.
      if(main_invoker.name !== childNode.name){
        let res = getJsonElementFromTwoNode(main_invoker, childNode, sequence, childNodeType=childNodeType);
        // if seqHashSet dose not have res as key, add to seqMap.
        // otherwise, do nothing.
        let seqKey = getStringFromEdge(res);
        if (!(seqHashSet.has(seqKey))){
          seqHashSet.add(seqKey);
          let next_number = number_generator.next()["value"];
          seqMap[next_number.toString()] = res;
        }
      }

      main_process(childNode, main_invoker, seqMap, seqHashSet, newSeq, duplicateCheck, MAX_DISTANCE=MAX_DISTANCE);
      return;
    });
  }

  let seqMap = new Object(null);
  let queue = new Queue();
  let number_generator = numbers();
  let ySet = new Set();
  queue.enqueue(ast);

  let check = 0;
  let checkList = new Set();

  // Transfer the scopeids to the tokens as well
  estraverse.traverse(ast, {
    enter : function (node) {
      if (node.type === "Identifier") {
        if (node.name !== undefined && node.name !== "undefined" && node.name !== "NaN" && node.name !== "Infinity") {

          if (node.scopeid !== undefined) {
            var index = rangeToTokensIndexMap[node.range + ""];
            var token = tokens[index];
            token.scopeid = node.scopeid;
          }
        }
      }
    }
  });

  // Build tag for element to infer or not to infer.
  estraverse.traverse(ast, {
    enter : function (node) {
      if (node.type === "Identifier") {
        if (node.name !== undefined && node.name !== "undefined" && node.name !== "NaN" && node.name !== "Infinity") {
          var index = rangeToTokensIndexMap[node.range + ""];
          var p = tokens[index - 1];
          if (p && p.type === "Punctuator" && p.value === ".") {
            node.isInfer = "element";
            return;
          }
          if (node.scopeid > 0) {
            node.isInfer = "id";
            return;
          }
        }
      }
      if (node.type === "Literal" | node.type === "ArrayExpression") {
        node.isInfer = "element";
      }
    }
  });


  while(n = queue.dequeue()){
    let parentNodeType = n.isInfer;

    // when node is not element or id.
    // not start main process
    // add child node to queue
    if(parentNodeType != "id"){
      let children = n.children;
      if(children){
        children.forEach( function(childNode){
          let range1 = childNode.range[0].toString();
          let range2 = childNode.range[1].toString();
          let rangeToken = range1 + DIVIDER + range2;

          if(!(checkList.has(rangeToken))){
            queue.enqueue(childNode);
            checkList.add(rangeToken);
          }
        });
      }
      continue
    }

    let initial_seq = "";
    // main process
    let duplicateCheck = new Set();
    let nodeName = n.scopeid + DIVIDER + n.name;
    ySet.add(nodeName);

    let seqHashSet = new Set();

    main_process(n, n, seqMap, seqHashSet, initial_seq, duplicateCheck);

    let children = n.children;
    if(children){
      children.forEach( function(childNode){
        let range1 = childNode.range[0].toString();
        let range2 = childNode.range[1].toString();
        let rangeToken = range1 + DIVIDER + range2;

        if(!(checkList.has(rangeToken))){
          queue.enqueue(childNode);
          checkList.add(rangeToken);
        }
      });
    }
  }

  seqMap["y_names"] = Array.from(ySet);
  return seqMap;
}

function extractNodeSequences(ast, tokens, rangeToTokensIndexMap, number, scopeParentMap){
  let sequences = [];

  // list of elements to infer or not to infer.
  var ids = [];
  var elements = [];

  function rangeContainCheck(par, chil){
    return chil[0] >= par[0] && chil[1] <= par[1];
  }

  function nodesBetweenTwoNode(x, y){
    // initialize
    let now_x = x;
    let now_y = y;
    let x_sequence = "";
    let y_sequence = "";

    while(1){
      xRange = now_x.range;
      yRange = now_y.range;

      if(xRange[0] == yRange[0] && xRange[1] == yRange[1]){
        break;
      }

      // xrange contains yrange
      if(rangeContainCheck(xRange, yRange)){
        try{
          now_y = getParent(now_y, ast);
        }
        catch(err){
          now_y = ast;
        }

        let add_token;
        // if nodeNameMap does not contain now_y.type, add to dic.
        if(!(now_y.type in nodeNameMap)){
          nodeNameMap[now_y.type] = String.fromCharCode(ascii_number);
          ascii_number += 1;
        }
        add_token = nodeNameMap[now_y.type];
        y_sequence = add_token + y_sequence;
      }
      // yrange contains xrange or else
      else{
        try{
          now_x = getParent(now_x, ast);
        }
        catch(err){
          now_x = ast;
        }

        let add_token;
        // if nodeNameMap does not contain now_y.type, add to dic.
        if(!(now_x.type in nodeNameMap)){
          nodeNameMap[now_x.type] = String.fromCharCode(ascii_number);
          ascii_number += 1;
        }
        add_token = nodeNameMap[now_x.type];
        x_sequence = x_sequence + add_token;
      }
    }

    x_sequence.slice(0, -1)
    let result = x_sequence + y_sequence;

    if(result.length >= 5){
      return null;
    }
    return result;
  }

  // Transfer the scopeids to the tokens as well
  let number_generator = numbers();

  estraverse.traverse(ast, {
    enter : function (node) {
      if (node.type === "Identifier") {
        if (node.name !== undefined && node.name !== "undefined" && node.name !== "NaN" && node.name !== "Infinity") {

          if (node.scopeid !== undefined) {
            var index = rangeToTokensIndexMap[node.range + ""];
            var token = tokens[index];
            token.scopeid = node.scopeid;
          }
        }
      }
    }
  });

  // Build list of element to infer or not to infer.
  estraverse.traverse(ast, {
    enter : function (node) {
      if (node.type === "Identifier") {
        if (node.name !== undefined && node.name !== "undefined" && node.name !== "NaN" && node.name !== "Infinity") {
          var index = rangeToTokensIndexMap[node.range + ""];
          var p = tokens[index - 1];
          if (p && p.type === "Punctuator" && p.value === ".") {
            elements.push(node);
            return;
          }
          if (node.scopeid > 0) {
            ids.push(node);
            return;
          }
        }
      }
      if (node.type === "Literal" | node.type === "ArrayExpression") {
        elements.push(node)
      }
    }
  });

  var seqMap = new Object(null);
  let y_set = new Set();

  for(let i=0; i < ids.length; i++){
    let x = ids[i];
    let xName = x.scopeid + DIVIDER + x.name;

    // add array of y names (to infer)
    y_set.add(xName);

    // extract sequences between two id
    for(let j=0; j < ids.length; j++){
      if(i==j) continue;
      let y = ids[j];

      // check scope relation
      if((scopeParentMap[x.scope.id].indexOf(y.scope.id) == -1) && (scopeParentMap[y.scope.id].indexOf(x.scope.id) == -1)){
        continue;
      }
      let yName = y.scopeid + DIVIDER + y.name;

      let seq = nodesBetweenTwoNode(x,y);
      // if seq is too long, null is returned
      if(seq === null) continue;

      let next_number = number_generator.next()["value"];
      let tmp = {"type":"var-var", "xName":x.name, "xScopeId":x.scopeid, "yName":y.name, "yScopeId":y.scopeid, "sequence": seq }
      seqMap[next_number.toString()] = tmp;
    }

    for(let j=0; j < elements.length; j++){
      let indexJ = j+ids.length;
      let y = elements[j]
      var name;
      // console.log(y)
      if(y.type === "Literal"){
        name = y["raw"];
      }
      else if(y.type === "ArrayExpression"){
        name = "Array";
      }
      else{
        name = y["name"];
      }

      let seq = nodesBetweenTwoNode(x,y);
      // if seq is too long, null is returned
      if(seq === null) continue;

      let tmp = {"type":"var-lit", "xName":x.name, "xScopeId":x.scopeid, "yName":name, "sequence": seq }
      let next_number = number_generator.next()["value"];
      seqMap[next_number.toString()] = tmp;
    }
  }
  seqMap["y_names"] = Array.from(y_set);
  return seqMap;
}

function extractSequences(ast, tokens, rangeToTokensIndexMap) {
  let sequences = [];

  function appendToken(arr, index, i) {
    var t, prev;
    if (i < 0) {
      arr.push("0START");
    } else if (i >= tokens.length) {
      arr.push("0END");
    } else if (i !== index) {
      t = tokens[i];
      if (t === undefined) {
        console.log(t);
      }
      prev = tokens[i - 1];
      if (t.type === "Identifier" && !(prev && prev.value === "." && prev.type === "Punctuator")) {
        if (t.hasOwnProperty("scopeid")) {
          arr.push("1ID:" + t.scopeid + ":" + t.value);
        } else {
          arr.push("1ID:-1:" + t.value);
        }
      } else if (!(t.value === "(" || t.value === ")" || t.value === ".")) {
        arr.push(t.value);
      }
    }
  }

  function appendVarUsage(node) {
    var index = rangeToTokensIndexMap[node.range + ""];
    var p = tokens[index - 1];
    if (p && p.type === "Punctuator" && p.value === ".") {
      return;
    }

    if (node.scopeid > 0) {
      var arr = [];
      var i, t, prev;
      for (i = index - 1; arr.length < WIDTH; i--) {
        appendToken(arr, index, i);
      }
      arr.reverse();
      arr.unshift(node.scope);
      arr.unshift(node.name);
      arr.unshift(node.scopeid);
      for (i = index + 1; arr.length < 3 + 2 * WIDTH; i++) {
        appendToken(arr, index, i);
      }
      sequences.push(arr);
    }
  }

  // Transfer the scopeids to the tokens as well

  estraverse.traverse(ast, {
    enter : function (node) {
      if (node.type === "Identifier") {
        if (node.name !== undefined && node.name !== "undefined" && node.name !== "NaN" && node.name !== "Infinity") {

          if (node.scopeid !== undefined) {
            var index = rangeToTokensIndexMap[node.range + ""];
            var token = tokens[index];
            token.scopeid = node.scopeid;
          }
        }
      }
    }
  });

  // Create the sequences

  estraverse.traverse(ast, {
    enter : function (node) {
      if (node.type === "Identifier") {
        if (node.name !== undefined && node.name !== "undefined" && node.name !== "NaN" && node.name !== "Infinity") {

          appendVarUsage(node);
        }
      }
    }
  });

  return sequences;
}

function writeSequences(sequences, outFile, fname, mode) {
  var seqMap = new Object(null);
  for (var i = 0; i < sequences.length; i++) {
    let sequence = sequences[i];
    var key = "" + sequence[0] + sequence[1];
    var val = seqMap[key];
    if (!val) {
      seqMap[key] = val = ["", sequence[1], sequence[0], sequence[2]];
    }
    for (var j = 3; j < sequence.length; j++) {
      var token = sequence[j] + "";
      var tokens = token.split(/(\s+)/);
      token = tokens[0];
      if (val[0].length > 0) {
        val[0] = val[0] + " ";
      }
      val[0] = val[0] + token;
    }
  }

  if (mode && mode === "recovery") {
    var testcases = [];
    var scopes = [];
    for (var k in seqMap) {
      if (seqMap.hasOwnProperty(k)) {
        scopes.push(seqMap[k][3]);
        testcases.push(fname.replace(/ /g,"_") + " 1ID:" + seqMap[k][2] + ":" + seqMap[k][1] + " " + seqMap[k][0]);
      }
    }

    return [testcases, scopes];

  } else {
    var logStream = fs.createWriteStream(outFile, {'flags': 'a'});
    for (var k in seqMap) {
      if (seqMap.hasOwnProperty(k)) {
        logStream.write(fname.replace(/ /g,"_") + " 1ID:" + seqMap[k][2] + ":" + seqMap[k][1] + " " + seqMap[k][0] + "\n");
      }
    }

    logStream.end();
  }
}

function recover(args, ast, testcases, scopes) {
  function isOk2Rename(origName, newName, scope) {
    // Check if any of the child scopes (including this) has a use of a variable called newName, belong
    // to this or a higher scope
    // Basic Idea : Don't shadow a variable
    return !(useStrictDirective && newName === "arguments") && !scope.alreadyUsed(newName, origName);
  }

  function rename(origName, newName, scope) {
    if (args.debug)
      console.log("Renaming " + origName + " to " + newName + " in " + scope.id);

    // For all the uses of this variable, mark that newName is being used in that scope
    scope.renameVar(origName, newName);
  }

  if (testcases.length === 0) {
    // Nothing to do. The program stays as is.
    return 0;
  }

  var useStrictDirective = true;

  // Extract Directives
  for (var i = 0; i < ast.body.length; i++) {
    if (HOP(ast.body[i], "directive")) {
      if (ast.body[i].directive === "use strict")
        useStrictDirective = true;
    }
  }

  // Send to the server
  var response = syncrequest('POST', 'http://' + args.ip + ":" + args.port,
    { json : { 'tests' : testcases}});

  if (response.statusCode === 200) {
    // res format : [prediction_arrays, the original names in the file, runtime]
    // prediction arrays format : array of arrays, each inner array containing 10 tuples
    // inner prediction tuple format : [probability, new name, index of name in the original array of names]
    var res = JSON.parse(response.body.toString('utf-8'));

    // Begin assignment of new names using a priority queue
    var queue = new pq({ comparator: function(a, b) { return b[0] - a[0]; }});

    var next2use = []; // Captures the number of names tried for each variable
    for (var i = 0; i < res[1].length; i++) {
      queue.queue(res[0][i][0]); // the first prediction tuple for each variable
      next2use.push(1);
    }

    var unk_ctr = 0;

    while (queue.length !== 0) {
      var elem =  queue.dequeue();
      var origIdx = elem[2];
      var origName = res[1][origIdx].split(':')[2];
      var newName = elem[1];
      var curScope = scopes[origIdx];
      if (origName === "arguments")
        continue;

      if (isOk2Rename(origName, newName, curScope)) {
        rename(origName, newName, curScope);
      } else {
        if (next2use[origIdx] >= 10) { // No more predictions left
          if (isOk2Rename(origName, origName, curScope)) { // This is needed, it's not trivial!
            rename(origName, origName, curScope);
          } else {
            rename(origName, "C2N_" + unk_ctr + "_" + origName, curScope);
            unk_ctr += 1;
          }
        } else {
          queue.queue(res[0][origIdx][next2use[origIdx]]);
          next2use[origIdx] += 1;
        }
      }
    }

    // Go over the AST and assign new names
    estraverse.traverse(ast, {
      enter : function(node) {
        if (node.type === "Identifier") {
          if (node.name && node.scope && node.scopeid) {
            if (node.scopeid > 0) {
              if (node.isFuncName)
                node.name = node.scope.getRenaming("$FUNC$" + node.name);
              else
                node.name = node.scope.getRenaming(node.name);

            }
          }
        }
      }
    });

    // All Done!
    return 0;

  } else {
    return -1;
  }

}

function processFile(args, fname, outDir, number) {
  try {
    if (!args.no_normalization)
      fname = fname.substr(0, fname.length-3) + ".normalized.js";

    var code = fs.readFileSync(fname, 'utf-8');
    var ast = esprima.parse(code, {tokens: true, range: true});
    var tokens = ast.tokens;

    // filename of pasth
    var fileNameIndex = fname.lastIndexOf("/") + 1;
    var filename = fname.substr(fileNameIndex);

    // Create token2index map
    var rangeToTokensIndexMap = new Object(null);
    for (var i = 0; i < tokens.length; i++) {
      rangeToTokensIndexMap[tokens[i].range + ""] = i;
    }

    // processAst(ast, rangeToTokensIndexMap)

    // Annotate nodes with scopes
    let scopeParentMap = new Object(null)
    scoper.addScopes2AST(ast, scopeParentMap);

    // make child-parent relation.
    makeChildParentRelation(ast);

    // Extract Sequences
    // let writeSeq = extractNodeSequences(ast, tokens, rangeToTokensIndexMap, number, scopeParentMap);
    globalSeqHashMapWrapper[number] = new HashMap();
    let writeSeq = newExtractNodeSequences(ast, tokens, rangeToTokensIndexMap, number, scopeParentMap);

    // Dump the sequences
    writeOnJson(writeSeq, outDir, number);

    console.log("[+] [" + success + "/" + failed + "] Processed file : " + fname);
    return 0;

  } catch (e) {
    if (args.debug)
      console.error(e.stack);
    console.log("[-] [" + success + "/" + failed + "] Failed to process file : " + fname);
    return 1;
  }
}

function recoverFile(args, fname, outFile) {
  try {
    var code = fs.readFileSync(fname, 'utf-8');
    var startTime = process.hrtime();
    var ast = esprima.parse(code, {tokens: true, range: true});
    var tokens = ast.tokens;

    // Create token2index map
    var rangeToTokensIndexMap = new Object(null);
    for (var i = 0; i < tokens.length; i++) {
      rangeToTokensIndexMap[tokens[i].range + ""] = i;
    }

    // Annotate nodes with scopes
    scoper.addScopes2AST(ast);

    // Extract Sequences
    var sequences = extractSequences(ast, tokens, rangeToTokensIndexMap);
    var res = writeSequences(sequences, null, fname, "recovery");
    var testcases = res[0];
    var scopes = res[1];
    var isFuncs = res[2];

    // Start Recovery
    recover(args, ast, testcases, scopes, isFuncs);
    var elapsedTime = process.hrtime(startTime);
    elapsedTime = elapsedTime[0] * 1000 + elapsedTime[1]/1000000;
    if (args.ext)
      args.outfile = fname.substr(0, fname.length-6) + args.ext;

    if (args.outfile.endsWith(".js")) {
      if (args.stats)
        fs.writeFileSync(args.outfile.substr(0, args.outfile.length-3) + ".timing.stats", "Time : " + elapsedTime);
      fs.writeFileSync(args.outfile, escodegen.generate(ast));
    } else {
      if (args.stats)
        console.log("Time : " + elapsedTime);
      console.log(escodegen.generate(ast));
    }

    console.log("[+] [" + success + "/" + failed + "] Processed file : " + fname);
    return 0;

  } catch (e) {
    console.log("[-] [" + success + "/" + failed + "] Failed to recover file : " + fname);
    console.error(e.stack);
    return -1;
  }
}

// merge Two object
function mergeJson(source, target){
  for(var index in source){
    // if index does not exist, initialize.
    if(!(index in target)){
      target[index] = [];
    }
    for(var val of source[index]){
      target[index].push(val);
    }
  }
}

// add json content to exist json file.
function writeOnJson(s, outDir, number){
  let res = JSON.stringify(s, null, '  ');
  let outFile = path.join(outDir, number.toString()+".json");
  fs.writeFileSync(outFile, res, {"flag":"w"});
}

var parser = new ArgumentParser({addHelp : true, description: 'Context2Name Client'});
parser.addArgument(
  ['--ip'],
  {
    help : 'IP Address of the server. Required in recovery mode.',
    defaultValue : '127.0.0.1'
  }
);

parser.addArgument(
  ['--port'],
  {
    help : 'Port for the server. Required in recovery mode',
    defaultValue : '8080'
  }
);

parser.addArgument(
  ['-l', '--listmode'],
  {
    action : 'storeTrue',
    help : 'Use input file as a list of files',
    defaultValue : false
  }
);

parser.addArgument(
  ['-f', '--file'],
  {
    help : 'File to work on',
    required : true,
    dest : 'inpFile'
  }
);

parser.addArgument(
  ['-d', '--debug'],
  {
    action : 'storeTrue',
    help : 'Debugging mode',
    defaultValue : false

  }
);

parser.addArgument(
  ['-r', '--recovery'],
  {
    action : 'storeTrue',
    help : 'Recovery Mode (Default = False)',
    defaultValue : false

  }
);

parser.addArgument(
  ['-s', '--stats'],
  {
    action : 'storeTrue',
    help : 'Collect relevant stats (Applicable only in recovery mode)',
    defaultValue : false

  }
);

parser.addArgument(
  ['--ext'],
  {
    action : 'store',
    //type : 'str',
    help : 'Extension to use for the recovered file (Default = null). Assumes that the filename passed ends in min.js',
    defaultValue : null
  }
);

parser.addArgument(
  ['-t', '--training-data'],
  {
    action : 'storeTrue',
    help : 'Training Mode. Creates the data to use for training. (Default = True)',
    defaultValue : true
  }
);

parser.addArgument(
  ['-w', '--width'],
  {
    action : 'store',
    type : 'int',
    help : 'Width of the contexts to use (Default : 5)',
    defaultValue : 5
  }
);

parser.addArgument(
  ['--no-normalization'],
  {
    action : 'storeTrue',
    help : "Don't use normalized versions of the input JS files in training data generation mode. Not recommended",
    defaultValue : false
  }
);

parser.addArgument(
  ['-a', '--append-mode'],
  {
    help : 'Append Mode. Useful while constructing training data.',
    required : false,
    action : 'storeTrue',
    defaultValue : false
  }
);

parser.addArgument(
  ['--outfile'],
  {
    help : 'Output File (Applicable only in training data extraction mode)',
    defaultValue : 'output.csv'
  }
);


parser.addArgument(
  ['--outdir'],
  {
    help : 'Output Directory(Applicable only in training data extraction mode)',
    defaultValue : 'output'
  }
);

parser.addArgument(
  ['-j', '--json'],
  {
    help : 'Json of Node-Char Map',
    dest : 'nodeMap'
  }
);

var args = parser.parseArgs();
if (!args.append_mode) {
  var logStream = fs.createWriteStream(args.outfile, {'flags': 'w'});
  logStream.end();
}

var WIDTH = args.width;

function existsFile(filename, callback) {
    fs.access(filename, "r", function (err, fd) {
        callback(!err || err.code !== "ENOENT");
    });
}

let use_map_name = args.nodeMap;

existsFile(use_map_name, function(result) {
  if (result) {
    fs.readFile(use_map_name, function read(err, data) {
      if (err) {
        throw err;
      }
      nodeNameMap = data;
    });
  }
});

if (args.recovery) {
  var success = 0;
  var failed = 0;
  if (args.listmode) {
    var readline = require('readline');

    var rl = readline.createInterface({
      input: fs.createReadStream(args.inpFile)
    });

    rl.on('line', function (line) {
      var s = recoverFile(args, line, args.outfile);
      if (s == 0) success += 1;
      else failed += 1;
    });

  } else {
    recoverFile(args, args.inpFile, args.outfile);
  }

} else {
  var success = 0;
  var failed = 0;
  if (args.listmode) {
    let number = 0;
    var readline = require('readline');

    var rl = readline.createInterface({
      input: fs.createReadStream(args.inpFile)
    });

    rl.on('line', function (line) {
      var s = processFile(args, line, args.outdir, number);
      number += 1;
      if (s == 0) success += 1;
      else failed += 1;
    });
  } else {
    processFile(args, args.inpFile, args.outfile);
  }
}
