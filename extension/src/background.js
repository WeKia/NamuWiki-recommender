chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    const res = {farewell: "well"};

    if (request.action === "Recommend"){
        var Docs = Recommendation(request.recentView, sendResponse);

        
    }
    return true;
});

async function Recommendation(recentView, sendResponse){
    console.log("processs start");

    const batch_size = 32;
    const idx = await GetSequence(recentView);

    const model_url = chrome.runtime.getURL('./model/model.json');
    const r = await fetch(model_url);
    const model = await r.arrayBuffer();

    console.log("Loading Model");

    try{
        const model = await tf.loadLayersModel(model_url);

        console.log("Model load complete!");

        const seq = await GetSequence(recentView);
        var seq_tensor = tf.tensor(seq, [1, 100]).tile([batch_size, 1]);
        
        console.log("Loading sequnces");
        
        var Scores = [];

        const loop_length = Math.ceil(379679/batch_size);

        for(var i=0; i < loop_length; i++){
            item_batch = null;
            if (i < loop_length - 1){
                item_batch = tf.range(batch_size * i, batch_size *(i + 1));
            }
            else {
                // if Batch is last batch
                seq_tensor =  tf.tensor(seq, [1, 100]).tile([(379679  % batch_size), 1]);
                item_batch = tf.range(batch_size * i, batch_size * i + (379679  % batch_size));
            }
            
            var output = model.predict([seq_tensor, item_batch]);

            Scores.push(...output.dataSync());
        }

        console.log(Scores);

        const Top10Docs = await GetTitles(argsort(Scores).slice(0, 10));
        console.log(Top10Docs);

        sendResponse(Top10Docs);
    }
    catch (e) {
        console.error(`failed to create inference session: ${e}`);
    }

    console.log("done");
};

async function GetSequence(recentView){
    var recent_idx = []

    const json_url = chrome.runtime.getURL('/data/title_idx.json');
    const r = await fetch(json_url);

    const title_idx = await r.json();

    for(var i=0; i < recentView.titles.length; i++) {
        idx = title_idx[recentView.titles[i]]

        if (typeof idx == "undefined" || idx == null){
            continue;
        }
        
        recent_idx.push(Number(idx));
    }

    const idx_length = recent_idx.length;

    //padding
    for(var i=0; i < 100 - idx_length; i++){
        recent_idx.push(0);
    }

    return recent_idx;
};

async function GetTitles(idx){
    var titles = []

    const json_url = chrome.runtime.getURL('/data/idx_title.json');
    const r = await fetch(json_url);

    const idx_title = await r.json();

    for(var i=0; i < idx.length; i++) {
        title = idx_title[idx[i]]
        
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