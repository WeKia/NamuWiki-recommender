function Recommend(){

    chrome.runtime.sendMessage({action:"Recommend", recentView: recentView}, function(response) {
        if(response === "Already Start!")
        {
            alert("작업이 이미 시작되었습니다!");
        }
    });

};
