function Recommend(){

    chrome.runtime.sendMessage({action:"Recommend", recentView: recentView}, function(response) {
        console.log(response);

        recDoc = {'titles' : []};

        for(var i=0;i < response.length; i++){
            recDoc.titles.push(response[i]);
        }

        chrome.storage.local.set({"Recommend" : recDoc}, function(){});
    });

};
