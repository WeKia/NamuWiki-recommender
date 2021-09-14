let recentView = {'titles' : []};
let recDoc = {'titles' : []};

let recDocEle;
let recentViewEle;

function AddSide(title_txt , btn_txt, onclick, key) {
  let base = document.createElement('div');
  let title  = document.createElement('h5');
  let innerbox = document.createElement('div');
  let tablebtn = document.createElement('a');

  base.id = 'recentView';
  base.className = 'c';
  base.className = 'basetable'

  title.className = 'tabletitle'
  title.innerText = title_txt;

  innerbox.className = 'innerbox';
  innerbox.id = 'RVinnerbox';
  
  tablebtn.className = 'tablebutton';
  tablebtn.innerText = btn_txt;
  tablebtn.onclick = onclick;

  base.appendChild(title);
  base.appendChild(innerbox);
  base.appendChild(tablebtn);

  document.body.appendChild(base);

  UpdateSide(base, key);

  return base;
}

function ResetRecentView()
{
  chrome.storage.local.remove("recentView",function(){
      recentView = {'titles' : []};
   });

  chrome.storage.local.remove("Recommend", function(){
      recDoc = {'titles' : []};
  });
}

function UpdateSide(Base, storage_key) {
  let div = Base.getElementsByClassName('innerbox')[0];

  div.innerHTML = '';

  while ( div.hasChildNodes() )
  {
    div.removeChild( div.firstChild );       
  }

  chrome.storage.local.get(storage_key, function(data){
    if(data.recentView)
    {
      recentView =  data.recentView;

      for(var i=0; i < recentView.titles.length; i++) {
        let a = document.createElement('a');
  
        a.className = 'innertext';
  
        a.innerText = recentView.titles[i];
        a.href = '/w/' + encodeURI(recentView.titles[i]);
  
        div.appendChild(a);
  
      }
    }

    if(data.Recommend)
    {
      recDoc =  data.Recommend;

      for(var i=0; i < recDoc.titles.length; i++) {
        let a = document.createElement('a');
  
        a.className = 'innertext';
  
        a.innerText = recDoc.titles[i];
        a.href = '/w/' + encodeURI(recDoc.titles[i]);
  
        div.appendChild(a);
  
      }
    }

  });
}

chrome.storage.onChanged.addListener((change, namespace) => {
  console.log("getChagned");

  UpdateSide(recentViewEle, "recentView");
  UpdateSide(recDocEle, "Recommend");
});

chrome.storage.local.get("recentView", function(data){
  if(data.recentView)
  {
    recentView =  data.recentView;
  }

  recentViewEle = AddSide('최근 방문', '[초기화]', ResetRecentView, "recentView");

});

chrome.storage.local.get("Recommend", function(data){
  if(data.Recommend)
  {
    recDoc =  data.Recommend;
  }

  recDocEle = AddSide('문서 추천', '추천 받기', Recommend, "Recommend");

});

// code from https://stackoverflow.com/questions/6390341/how-to-detect-if-url-has-changed-after-hash-in-javascript/41825103#41825103 @Alburkerk
// I think since NamuWiki use AMP (Accelerated Mobile Pages), moving to other document doesn't load all contents (ex. sidebar, menu)
// Therfore some events (popstatechange, onhashchange, change) doesn't fires.
// To overcome such issue, we need to check url is changed using setTimeout
// But it may cause lag..

var oldHref = '';
function listen(currentHref) {
  if (currentHref != oldHref) {
    // Do your stuff here
    if(currentHref.indexOf("namu.wiki/w/")> -1)
    {

      const docu_name =  decodeURIComponent(currentHref.split("namu.wiki/w/")[1].split('?')[0].split('#')[0]);

      if ((recentView.titles.indexOf(docu_name) == -1) && (docu_name.substring(0, 3) != "분류:") )
      {
        recentView.titles.unshift(docu_name);

        if (recentView.titles.length > 20){
          recentView.titles.pop();
        }
    
        chrome.storage.local.set({"recentView" : recentView}, function(){});
      }
    }
  }

  oldHref = window.location.href;
  setTimeout(function () {
    listen(window.location.href);
  }, 500);
}

document.addEventListener("DOMContentLoaded", function(event) { 
  
  let side = document.getElementsByTagName('aside')[0];

  side.prepend(recDocEle);
  side.prepend(recentViewEle);

  listen(window.location.href);
});

document.addEventListener('load', function(envet){
  AddPageChangeEvent();
});
