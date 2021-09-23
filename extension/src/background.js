var running = false;

class NormLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
    }

    computeOutputShape(inputShape) { return inputShape; }
    
    call(input) {
        return tf.tidy(() => {return tf.div(input[0], tf.expandDims(tf.norm(input[0], 'euclidean', 1), 1))});
    }

    static get className() {
        return 'NormLayer';
    }
};

class MeanPool extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.supportsMasking = true;
    }

    call(x, args){
        if (Object.keys(args).length > 0){
            
            return tf.tidy(() =>{
                var mask = tf.cast(args.mask, 'float32')
                mask = tf.expandDims(mask, 2)
                x = tf.mul(x[0], mask)
                
                return tf.div(tf.sum(x, 1), tf.sum(mask, 1))})
        }
        else
            return tf.mean(x, 1)
    }
        

    computeOutputShape(input_shape){
        // remove temporal dimension
        return [input_shape[0], input_shape[2]]
      }
      
    static get className() {
        return 'MeanPool';
    }
}

tf.serialization.registerClass(NormLayer);
tf.serialization.registerClass(MeanPool);

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    const res = {farewell: "well"};

    if (request.action === "Recommend"){

        if (!running){
            var Docs = Recommendation(request.recentView, sendResponse);
        }
        else {
            sendResponse("Already Start!");
        }

    }
    return true;
});

async function Recommendation(recentView, sendResponse){
    console.log("processs start");

    running = true;

    var batch_size = 64;
    chrome.storage.local.get("batchVal", ({ batchVal }) => {
        if(batchVal) {
            console.log(batchVal);
            batch_size = batchVal;
        }
    });
    const idx = await GetSequence(recentView);

    const model_url = chrome.runtime.getURL('./model/model.json');
    const r = await fetch(model_url);
    const model = await r.arrayBuffer();

    console.log("Loading Model");

    try{

        tf.engine().startScope()
        const model = await tf.loadLayersModel(model_url);

        console.log("Model load complete!");

        const seq = await GetSequence(recentView);
        var seq_tensor = tf.tensor(seq).tile([batch_size, 1]);
        
        console.log("Loading sequnces");
        
        var Scores = [];

        const loop_length = Math.ceil(379440/batch_size);

        const tenth = Math.floor(loop_length/10);

        for(var i=0; i < loop_length; i++){
            item_batch = null;
            if (i < loop_length - 1){
                item_batch = tf.range(batch_size * i, batch_size *(i + 1));
            }
            else {
                // if Batch is last batch
                seq_tensor =  tf.tensor(seq).tile([(379440  % batch_size), 1]);
                item_batch = tf.range(batch_size * i, batch_size * i + (379440  % batch_size));
            }
            
            var output = model.predict([seq_tensor, item_batch]);

            Scores.push(...output.dataSync());

            if((i+1) % tenth == 0) {
                console.log("Process " + (Math.floor((i +1) / tenth) * 10) + "% Success");
            }
        }

        const Top10Docs = await GetTitles(argsort(Scores).reverse().slice(1, 21));

        recDoc = {'titles' : []};

        for(var i=0;i < Top10Docs.length; i++){
            recDoc.titles.push(Top10Docs[i]);
        }

        chrome.storage.local.set({"Recommend" : recDoc}, function(){});

        sendResponse("Done!");
    }
    catch (e) {
        console.error(`failed to create inference session: ${e}`);
    }

    tf.engine().endScope()
    console.log("done");
    running = false;
};

async function GetSequence(recentView){
    var recent_idx = []


    var title_idx = null;
    try{
        const json_url = chrome.runtime.getURL('/data/title_idx.json');
        const r = await fetch(json_url);
        title_idx = await r.json();
    }
    catch (e) {
        console.error('File title_idx.json is not found!');
    }

    var setRecentVal = 20;

    chrome.storage.local.get("recentVal", ({ recentVal }) => {
        if(recentVal) {
            console.log(recentVal);
            setRecentVal = recentVal;
        }
    });

    const loop = Math.min(setRecentVal, recentView.titles.length);

    for(var i=0; i < loop; i++) {
        idx = title_idx[recentView.titles[i]]

        if (typeof idx == "undefined" || idx == null){
            continue;
        }
        
        recent_idx.push(Number(idx) + 1);
    }

    const idx_length = recent_idx.length;

    //padding
    for(var i=0; i < 20 - idx_length; i++){
        recent_idx.push(0);
    }

    return [recent_idx];
};

async function GetTitles(idx){
    var titles = []

    const json_url = chrome.runtime.getURL('/data/idx_title.json');
    const r = await fetch(json_url);

    const idx_title = await r.json();

    for(var i=0; i < idx.length; i++) {
        title = idx_title[idx[i] - 1]
        
        titles.push(title);
    }

    return titles;
}

// Code from https://titanwolf.org/Network/Articles/Article?AID=135aa377-6989-4a7c-8da9-6eb73ef33086#gsc.tab=0
function argsort(array) {

    var arrayObject = [];

    for(var i=0; i < array.length; i++) {
        arrayObject.push({value : array[i], idx : i});
    }

    arrayObject.sort((a, b) => {

        if (a.value < b.value) {

            return -1;

        }

        if (a.value > b.value) {

            return 1;

        }

        return 0;

    });

    const argIndices = arrayObject.map(data => data.idx);

    return argIndices;
 };