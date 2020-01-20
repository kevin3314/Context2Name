var scoper = require(__dirname + '/scoper.js');
var pq = require('js-priority-queue');

var fs = require('fs');
var esprima = require('esprima');
var estraverse = require('estraverse');
var getParent = require('estree-parent')
var HashMap = require('hashmap');
var HashTable = require('hashtable');

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

function getNodeTokenOfSequence(node, nodeNameMap){
  let nodetype;
  // if nodenamemap does not contain node.type, add to dic.
  if(node.type == "BinaryExpression"){
    nodetype = node.operator;
  }
  else{
    nodetype = node.type;
  }

  if(!(nodetype in nodeNameMap)){
    nodeNameMap[nodetype] = String.fromCharCode(ascii_number);
    ascii_number += 1;
  }
  let add_token = nodeNameMap[nodetype];

  return add_token;
}

function reverseString(str) {
    var splitString = str.split(""); // var splitString = "hello".split("");
    var reverseArray = splitString.reverse(); // var reverseArray = ["h", "e", "l", "l", "o"].reverse();
    var joinArray = reverseArray.join(""); // var joinArray = ["o", "l", "l", "e", "h"].join("");
    return joinArray;
}

function extractNodeSequences(ast, tokens, rangeToTokensIndexMap, number, scopeParentMap){
  function getRangeToken(node){
    return node.range + "";
  }

  function getIsId(node){
    return node.isInfer == "id";
  }

  function updateHashTable(node, hashTable, seqHashSet, rangeToTokensIndexMap, rangeToNodeNameMap, seqMap, tokens, y_List, number_generator, MAX_DISTANCE=10){
    let nodeHashSet = new Set();
    let nodeIsId = getIsId(node);

    // get token correspodnigNodeNamee
    let newToken = getNodeTokenOfSequence(node, nodeNameMap);

    let nodeToken = getRangeToken(node);

    // Get N_e
    let epsilons;
    if (node.children){
      epsilons = node.children;
    }
    else{
      // this node is leaf, break.
      epsilons = [];
      // return [];
    }

    if(node.parent){
      epsilons.push(node.parent);
    }

    hashTable[nodeToken] = new Object(null);

    // Update Map from N_e, T
    epsilons.forEach(function(epsilon){
      let epsilonIsId = getIsId(epsilon);
      let epsilonToken = getRangeToken(epsilon);
      if(!hashTable.hasOwnProperty(epsilonToken)){
        hashTable[epsilonToken] = new Object(null);
      }

      // Update Map from N_e
      hashTable[epsilonToken][nodeToken] = ["", epsilonIsId];

      // Update Map from T: add N_e
      hashTable[nodeToken][epsilonToken] = ["", nodeIsId];
    });

    // Update Map fron N_n
    epsilons.forEach(function(epsilon){
      let epsilonToken = getRangeToken(epsilon);

      // Get N_n from hashTable[epsilon]
      let N_nList = Object.keys(hashTable[epsilonToken]);

      // iterate each N_n
      N_nList.forEach(function(N_n){
        if(nodeHashSet.has(N_n)){ return; }
        nodeHashSet.add(N_n);

        // Update hashTable[N_n]
        let seqAndBool = hashTable[N_n][epsilonToken];
        let N_nIsId = seqAndBool[1];

        if(seqAndBool[0].length >= MAX_DISTANCE) { return; }
        let N_nToNodeSeq = seqAndBool[0] + newToken;
        hashTable[N_n][nodeToken] = [N_nToNodeSeq, seqAndBool[1]];

        let reversed_seq = reverseString(seqAndBool[0]);
        let nodeToN_nSeq = newToken + reversed_seq;
        hashTable[nodeToken][N_n] =  [nodeToN_nSeq, nodeIsId];

        let edges = [];

        // write on output json.
        // element-id or id-id should be handled, otherwise pass.
        if(!nodeIsId && !N_nIsId) { return; }

        let i_1 = rangeToTokensIndexMap[nodeToken];
        let token_1 = tokens[i_1];
        let i_2 = rangeToTokensIndexMap[N_n];
        let token_2 = tokens[i_2];

        // id-id
        if(nodeIsId && N_nIsId){
          // get node's varible index
          let node1Name = token_1.scopeid + DIVIDER + token_1.value;
          let node2Name = token_2.scopeid + DIVIDER + token_2.value;

          let node1Index = yList.indexOf(node1Name);
          let node2Index = yList.indexOf(node2Name);

          edge1 = {
            "type":"var-var",
            "xName":token_1.value,
            "xScopeId":token_1.scopeid,
            "xIndex": node1Index,
            "yName":token_2.value,
            "yScopeId":token_2.scopeid,
            "yIndex":node2Index,
            "sequence": nodeToN_nSeq
          };

          edge2 = {
            "type":"var-var",
            "xName":token_2.value,
            "xScopeId":token_2.scopeid,
            "xIndex": node2Index,
            "yName":token_1.value,
            "yScopeId":token_1.scopeid,
            "yIndex": node1Index,
            "sequence": N_nToNodeSeq
          };
          edges = [edge1, edge2];
        }

        // element-id
        else{
          token1NodeName = rangeToNodeNameMap[nodeToken];
          token2NodeName = rangeToNodeNameMap[N_n];

          // token_1 or token_2 is not token; only block, statement
          if(!token_1 || !token_2) { return; }

          // token_1 or token_2 is not element/variable
          if(!token1NodeName || !token2NodeName) { return; }

          let xName, xScopeId, xIndex, yName, seq;
          if (nodeIsId){
            // get node's varible index
            let node1Name = token_1.scopeid + DIVIDER + token_1.value;

            let node1Index = yList.indexOf(node1Name);
            let node2Index = yList.indexOf(token2NodeName);

            xName = token_1.value;
            xScopeId = token_1.scopeid;
            xIndex = node1Index;
            yName = token2NodeName;
            yIndex = node2Index;
            seq = nodeToN_nSeq;
          }
          else{
            // get node's varible index
            let node2Name = token_2.scopeid + DIVIDER + token_2.value;

            let node1Index = yList.indexOf(token1NodeName);
            let node2Index = yList.indexOf(node2Name);

            xName = token_2.value;
            xScopeId = token_2.scopeid;
            xIndex = node2Index;
            yName = token1NodeName;
            yIndex = node1Index;
            seq = N_nToNodeSeq;
          }

          edge = {
            "type":"var-lit",
            "xName":xName,
            "xScopeId":xScopeId,
            "xIndex": xIndex,
            "yName":yName,
            "yIndex": yIndex,
            "sequence": seq
          };
          edges = [edge];
        }

        // add edge to seqMap
        edges.forEach(function(edge){
          let seqKey = getStringFromEdge(edge);
          if (!(seqHashSet.has(seqKey))){
            seqHashSet.add(seqKey);
            let next_number = number_generator.next()["value"];
            seqMap[next_number.toString()] = edge;
          }
        });

      });
    });
  }

  let seqMap = new Object(null);
  let seqHashSet = new Set();
  let queue = new Queue();
  let number_generator = numbers();

  let yList;
  let variableList = [];
  let variableSet = new Set();
  let knownList = [];
  queue.enqueue(ast);

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

  // Build rangeToNodeNameMap
  let rangeToNodeNameMap = new Object(null);

  // Build tag for element to infer or not to infer.
  estraverse.traverse(ast, {
    enter : function (node) {
      if (node.type === "Identifier") {
        if (node.name !== undefined && node.name !== "undefined" && node.name !== "NaN" && node.name !== "Infinity") {
          var index = rangeToTokensIndexMap[node.range + ""];
          var p = tokens[index - 1];
          if (p && p.type === "Punctuator" && p.value === ".") {
            rangeToNodeNameMap[node.range + ""] = node["name"];
            node.isInfer = "element";
            if(!knownList.includes(node["name"])){
              knownList.push(node["name"]);
            }
            return;
          }

          if (node.scopeid > 0) {
            let nodeName = node.scopeid + DIVIDER + node.name;
            rangeToNodeNameMap[node.range + ""] = nodeName;
            node.isInfer = "id";
            if(!variableList.includes(nodeName)){
              variableList.push(nodeName);
            }
            if(!variableSet.has(nodeName)){
              variableSet.add(nodeName);
            }
            return;
          }
        }
      }

      if (node.type === "Literal" || node.type === "ArrayExpression") {
        let index = rangeToTokensIndexMap[node.range + ""];

        let name;
        if(node.type === "Literal"){
          name = node["raw"];
        }
        else if(node.type === "ArrayExpression"){
          name = "Array";
        }
        else{
          name = node["name"];
        }
        rangeToNodeNameMap[node.range + ""] = name;
        if(!knownList.includes(name)){
          knownList.push(name);
        }

        node.isInfer = "element";
      }
    }
  });

  yList = variableList.concat(knownList);

  // let hashTable = new HashTable();
  let hashTable = new Object(null);

  estraverse.traverse(ast, {
    enter : function (node) {
      updateHashTable(node, hashTable, seqHashSet, rangeToTokensIndexMap, rangeToNodeNameMap, seqMap, tokens, yList, number_generator);
      }
  });

  seqMap["y_names"] = yList;
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

    // Extract Sequences
    globalSeqHashMapWrapper[number] = new HashMap();
    let writeSeq = extractNodeSequences(ast, tokens, rangeToTokensIndexMap, number, scopeParentMap);

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

nodeNameMap = JSON.parse(fs.readFileSync('simplified_map.json', 'utf8'));

console.log(nodeNameMap);

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
