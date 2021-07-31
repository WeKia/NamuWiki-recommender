let recentView = {'titles' : []};

let recentViewEle;

function AddSide() {
  let base = document.createElement('div');
  let title  = document.createElement('h5');
  let innerbox = document.createElement('div');
  let tablebtn = document.createElement('a');

  base.id = 'recentView';
  base.className = 'c';
  base.setAttribute('data-v-2cd3b089', '');
  base.setAttribute('data-v-014f19d0', '');

  //title.className = 'tabletitle'
  title.setAttribute('data-v-2cd3b089', '')
  title.innerText = "최근 방문";

  //innerbox.className = 'innerbox';
  innerbox.setAttribute('data-v-2cd3b089', '');
  innerbox.id = 'RVinnerbox';
  
  //tablebtn.className = 'tablebutton';
  tablebtn.setAttribute('data-v-2cd3b089', '');
  tablebtn.innerText = '[초기화]';
  tablebtn.onclick = ResetRecentView;

  base.appendChild(title);
  base.appendChild(innerbox);
  base.appendChild(tablebtn);

  document.body.appendChild(base);

  UpdateSide();

  return base;
}

function ResetRecentView()
{
  chrome.storage.local.remove("recentView",function(){
      recentView = {'titles' : []};
   });
}

function UpdateSide() {
  let div = document.getElementById('RVinnerbox');

  div.innerHTML = '';

  while ( div.hasChildNodes() )
  {
    div.removeChild( div.firstChild );       
  }

  chrome.storage.local.get("recentView", function(data){
    if(data.recentView)
    {
      recentView =  data.recentView;

      for(var i=0; i < recentView.titles.length; i++) {
        let a = document.createElement('a');
  
        //a.className = 'innerText';
        a.setAttribute('data-v-2cd3b089', '');
        a.setAttribute('class', '');
  
        a.innerText = recentView.titles[i];
        a.href = '/w/' + encodeURI(recentView.titles[i]);
  
        div.appendChild(a);
  
      }
    }

  });
}

chrome.storage.onChanged.addListener((change, namespace) => {
  UpdateSide();
});

chrome.storage.local.get("recentView", function(data){
  if(data.recentView)
  {
    recentView =  data.recentView;
  }

  recentViewEle = AddSide();

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

      if (recentView.titles.indexOf(docu_name) == -1)
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
  
  let side = document.getElementById('tchika770b3753').parentElement;

  side.prepend(recentViewEle);

  listen(window.location.href);
});

document.addEventListener('load', function(envet){
  AddPageChangeEvent();
});
