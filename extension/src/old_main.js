let blockedUsers = {'ids' : [], 'nicks': []};
let colorUsers = {'ids' : [], 'nicks': []};

document.addEventListener("DOMContentLoaded", function(event) { 
  let curpage = window.location.href;
  
  if(curpage.indexOf("ArticleRead")> -1)
  {
    initializeBlocks2();
    console.log("게시글!");
  }
  else if(curpage.indexOf('search.memberid') > -1)
  {
    console.log("유저정보!");
  }
  else if(curpage.indexOf('ArticleList') > -1)
  {
    initializeBlocks();
    setTimeout(function(){
      refresh();
    }, 3000);
  }
  else console.log(curpage);
  setTimeout(function(){
    getNewsActivity();
  }, 3000);
});

function initializeBlocks()
{
  chrome.storage.local.get("blockedUsers", function(data){
    if(data.blockedUsers)
    {
      blockedUsers =  data.blockedUsers;
    }
    chrome.storage.local.get("colorUsers", function(data){
      if(data.colorUsers)
      {
        colorUsers = data.colorUsers;
      }
      
      modification();
    })
  });
}

function initializeBlocks2()
{
  chrome.storage.local.get("blockedUsers", function(data){
    if(data.blockedUsers)
    {
      blockedUsers =  data.blockedUsers;
    }
    chrome.storage.local.get("colorUsers", function(data){
      if(data.colorUsers)
      {
        colorUsers = data.colorUsers;
      }
      modificationArticle();
    })
  });
}

function modificationArticle()
{
  let pnicks = document.getElementsByClassName('p-nick');

  for(var i = 0; i < pnicks.length; i++)
  {
    let id = "";
    let nick = "";

    if(i == 0 ||  pnicks.length - 3 < i )
    {
      id = pnicks[i].children[0].getAttribute('onclick').match(/'(.*?)'/g)[0].replace(/'/g, "");
      nick = pnicks[i].children[0].getAttribute('onclick').match(/'(.*?)'/g)[1].replace(/'/g, "");
    }
    else
    {
      let href = pnicks[i].parentElement.children[0].children[0].children[0].getAttribute("href");
      id = href.slice(href.indexOf("memberid=") + "memberid=".length);
      nick = pnicks[i].children[0].innerHTML;
    }
    
    pnicks[i].children[0].addEventListener('click', function(){addBlockButton(id, nick);});

    if(blockedUsers.ids.indexOf(id)> -1 && i > 0)
    {
      let ul = document.getElementsByClassName("cmlist")[0];
      let li = pnicks[i].parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.parentNode;
      li.setAttribute('style', 'display : none');
      let newli = document.createElement('li');
      newli.innerHTML = '<div class="comm_cont"><div class="h"><p class="comm m-tcol-c"><span class="comm_body" style="cursor:pointer;">차단된 사용자입니다.</span></p></div></div>';

      ul.insertBefore(newli, li);
      newli.appendChild(li);
      newli.onclick = function(){
        newli.setAttribute('style', 'display : none');
        li.removeAttribute('style');
        ul.insertBefore(li, newli);
      };
    }
  }
}

function modification()
{
  let url = window.location.href;
  let listType = url[url.search('boardtype=') + 10];
  let pnicks = document.querySelectorAll('td.p-nick');

  for(var i = 0; i < pnicks.length; i++)
  {
    let id = pnicks[i].children[0].getAttribute('onclick').match(/'(.*?)'/g)[0].replace(/'/g, "");
    let nick = pnicks[i].children[0].getAttribute('onclick').match(/'(.*?)'/g)[1].replace(/'/g, "");
    pnicks[i].children[0].addEventListener('click', function(){addBlockButton(id, nick);});


    if(colorUsers.ids.indexOf(id)> -1)
    {
      if (listType == "L")
      {
        let tr = pnicks[i].parentNode.parentNode.parentNode.parentElement.parentNode.parentElement;
        tr.setAttribute('style', 'background-color: rgba( 145, 215, 149, 0.5 );');
      }
      else if (listType == "C")
      {
        let li = pnicks[i].parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.parentNode;
        li.setAttribute('style', 'background-color: rgba( 145, 215, 149, 0.5 );');
      }
      else if (listType == "M")
      {
        let li = pnicks[i].parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.parentNode;
        li.setAttribute('style', 'background-color: rgba( 145, 215, 149, 0.5 );');
      }
      else if (listType == "I") 
      {
        let li = pnicks[i].parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.parentNode;
        li.setAttribute('style', 'background-color: rgba( 145, 215, 149, 0.5 );');
      }
    }
    if(blockedUsers.ids.indexOf(id)> -1)
    {
      if (listType == "L")
      {
        let tbody = pnicks[i].parentNode.parentNode.parentNode.parentElement.parentNode.parentElement.parentElement;
        let tr = pnicks[i].parentNode.parentNode.parentNode.parentElement.parentNode.parentElement;
        tr.setAttribute('style', 'display : none');
        let newtr = document.createElement('tr');
        let newtd = document.createElement('td');
        newtr.innerHTML = '<td colspan="2" class="td_article"><div class="board-list"><div class="inner_list"><a class="article" style="cursor:pointer;">차단된 사용자입니다.</a></div></div></td>';
  
        //tbody.insertBefore(newtr, tr);
        //newtr.appendChild(tr);
        newtr.onclick = function(){
          newtr.setAttribute('style', 'display : none');
          tr.removeAttribute('style');
          tbody.insertBefore(tr, newtr);
        };
      }
      else if (listType == "C")
      {
        let ul = document.getElementsByClassName('article-movie-sub')[0];
        let li = pnicks[i].parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.parentNode;
        li.setAttribute('style', 'display : none');

        let newli = document.createElement('li');
        newli.innerHTML = '<div class="card_area"><div class="con"><div class="con_top"><div class="tit_area"><a class="tit" style="cursor:pointer;"><span class="inner"><strong>차단된 사용자입니다.</strong></span></a></div></div></div></div>';

        //ul.insertBefore(newli, li);
        //newli.appendChild(li);
        newli.onclick = function(){
          newli.setAttribute('style', 'display : none');
          li.removeAttribute('style');
          ul.insertBefore(li, newli);
        };
      }
      else if (listType == "M")
      {
        let ul = document.getElementsByClassName('article-album-movie-sub')[0];
        let li = pnicks[i].parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.parentNode;
        li.setAttribute('style', 'display : none');

        let newli = document.createElement('li');
        newli.innerHTML = '<dl><dt class="tit_area"><a class="tit" style="cursor:pointer;"><span class="inner"><span clss="tit_txt">차단된 사용자입니다.</span></span></a></dt></dl>';

        //ul.insertBefore(newli, li);
        //newli.appendChild(li);
        newli.onclick = function(){
          newli.setAttribute('style', 'display : none');
          li.removeAttribute('style');
          ul.insertBefore(li, newli);
        };
      }
      else if (listType == "I") 
      {
        let ul = document.getElementsByClassName('article-album-sub')[0];
        let li = pnicks[i].parentNode.parentNode.parentNode.parentNode.parentNode.parentNode.parentNode;
        li.setAttribute('style', 'display : none');

        let newli = document.createElement('li');
        newli.innerHTML = '<dl><dt><a class="tit" style="cursor:pointer;"><span class="inner"><span clss="ellipsis">차단된 사용자입니다.</span></span></a></dt></dl>';

        //ul.insertBefore(newli, li);
        //newli.appendChild(li);
        newli.onclick = function(){
          newli.setAttribute('style', 'display : none');
          li.removeAttribute('style');
          ul.insertBefore(li, newli);
        };
      }
    }
  }
}

function getNewsActivity()
{
  chrome.storage.local.get("actrefreshCheck", function(ref)
  {
    if(ref.actrefreshCheck == null) return;
    
    if(ref.actrefreshCheck)
    {
      let URI = 'https://cafe.naver.com/MyNewsActivityListAjax.nhn';
      getPage(URI, function(data){
      notification(data);
      }, false);
    }
  });
  setTimeout(function() {
    getNewsActivity();
  }, 3000);
  
}

function refresh() 
{
  chrome.storage.sync.get("refreshCheck", function(refreshdata) {
      if(refreshdata == null) return;
      let URI = window.location.href
      getPage(URI, function(data){
          let newhtml = document.createElement('html');
          newhtml.innerHTML = data;
    
          if(refreshdata.refreshCheck)
          {
            let currentDocument = document.getElementsByClassName("article-board m-tcol-c")[1];
            let newDocument = newhtml.getElementsByClassName("article-board m-tcol-c")[1];
            if(currentDocument && newDocument)
            {
              currentDocument.outerHTML = newDocument.outerHTML;
            }
            modification();
          }
          //console.log(data);
      }, true);
      
  });
  setTimeout(function() {
    refresh();
  }, 3000);
}

function notification(data)
{
  chrome.storage.local.get('msgKey', function(datas)
  {
      let newhtml = document.createElement('html');
      newhtml.innerHTML = data;

      lis = newhtml.getElementsByTagName("li");


      for(var i = 0; i < lis.length; i ++)
      {
        cafeid = lis[i].getAttribute("data-cafeid");
        if (cafeid != "19543191")
        {
          console.log("not lolkor activity");
          continue;
        }
        
        newmsgKey = lis[i].getAttribute("data-messageKey");
        if(datas.msgKey == newmsgKey)
        {
          return;
        }

        comment = lis[i].getElementsByClassName("cc_mynews_item")[0].children[1].children[1].innerText;
        commnet = comment.replace(/(^\s+|\s+$)/g,'');
        icon = lis[i].getElementsByClassName("cc_mynews_item")[0].children[0].children[0].getAttribute("src");
        url = lis[i].getElementsByClassName("cc_mynews_item")[0].getAttribute("data-url");

        var opt = {
          body: comment,
          icon: icon
        };

        if(Notification.permission != 'granted')
        {
          Notification.requestPermission(function (permission) {
            // If the user accepts, let's create a notification
            if (permission == "granted") {
              var notification = new Notification("새로운 댓글 알림", opt);
            }
          });
        }
        else
        {
          var notification = new Notification("새로운 댓글 알림", opt);
        }
        
        notification.onclick = function(){
          window.open(url);
        }
        chrome.storage.local.set({"msgKey" : newmsgKey}, function(){});
        return;
      }
      
  });
}

function getPage(url, _callback, encode)
{
  var xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function() { // 요청에 대한 콜백
    if (xhr.readyState === xhr.DONE) { // 요청이 완료되면
      if (xhr.status === 200 || xhr.status === 201) {
        _callback(xhr.responseText);
      } else {
        console.error("ERROR Occured! " + xhr.responseText);
      }
    }
  };
  xhr.open('GET', url); // 메소드와 주소 설정
  if(encode){
    xhr.overrideMimeType("text/html;charset=euc-kr");
  }
  xhr.send(); // 요청 전송 
}

function blockUser(id, nick)
{
  if(blockedUsers.ids.indexOf(id) > -1)
  {
    if(confirm("정말로 " + nick+"("+ id +")의 차단을 해제 하시겠습니까?"))
    {
      blockedUsers.ids.splice(blockedUsers.ids.indexOf(id), 1);
      blockedUsers.nicks.splice(blockedUsers.nicks.indexOf(nick), 1);
      chrome.storage.local.set({"blockedUsers" : blockedUsers}, function(){});
    }
  }
  else
  {
    if(confirm("정말로 " + nick+"("+ id +")을 차단하시겠습니까?"))
    {
      blockedUsers.ids.push(id);
      blockedUsers.nicks.push(nick);

      chrome.storage.local.set({"blockedUsers" : blockedUsers}, function(){});
    }
  }
}

function coloringUsers(id, nick)
{
  if(colorUsers.ids.indexOf(id) > -1)
  {
    if(confirm("정말로 " + nick+"("+ id +")의 색을 알림을 해제 하시겠습니까?"))
    {
      colorUsers.ids.splice(colorUsers.ids.indexOf(id), 1);
      colorUsers.nicks.splice(colorUsers.nicks.indexOf(nick), 1);
      chrome.storage.local.set({"colorUsers" : colorUsers}, function(){});
    }
  }
  else
  {
    if(confirm("정말로 " + nick+"("+ id +")을 색을 알림하시겠습니까?"))
    {
      colorUsers.ids.push(id);
      colorUsers.nicks.push(nick);

      chrome.storage.local.set({"colorUsers" : colorUsers}, function(){});
    }
  }
}

function addBlockButton(i ,n)
{
  setTimeout(function(){
  let usrInform = document.getElementsByClassName('perid-layer')[0];

  if (usrInform.children[0].children.length > 5)
    return;
  let blockli = document.createElement('li');
  let blocka  = document.createElement('a');
  let blockspan = document.createElement('span');

  if(blockedUsers.ids.indexOf(i) > -1)
   blockspan.innerText = '차단해제';
  else
    blockspan.innerText = '차단하기';
  blocka.appendChild(blockspan);
  blocka.setAttribute("style", 'cursor:pointer;');
  blocka.onclick = function() { blockUser(i, n);};
  blockli.appendChild(blocka);
  usrInform.children[0].appendChild(blockli);

  let colorli = document.createElement('li');
  let colora  = document.createElement('a');
  let colorspan = document.createElement('span');

  if(colorUsers.ids.indexOf(i) > -1)
    colorspan.innerText = '알림해제';
  else
    colorspan.innerText = '색을알림';
  colora.appendChild(colorspan);
  colora.setAttribute("style", 'cursor:pointer;');
  colora.onclick = function() { coloringUsers(i, n);};
  colorli.appendChild(colora);
  usrInform.children[0].appendChild(colorli);
  }, 300);
}