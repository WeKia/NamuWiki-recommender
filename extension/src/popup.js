// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

'use strict';

let refreshCheck = document.getElementById('refreshCheck');
let ActrefreshCheck = document.getElementById('ActrefreshCheck');

chrome.storage.sync.get("refreshCheck", function(data) {
  if (data.refreshCheck != null)
    refreshCheck.checked = data.refreshCheck;
});

refreshCheck.onclick = function () {
  chrome.storage.sync.set({"refreshCheck" : refreshCheck.checked}, function(){});
};

chrome.storage.local.get("actrefreshCheck", function(data) {
  if (data.actrefreshCheck != null)
    ActrefreshCheck.checked = data.actrefreshCheck;
});

ActrefreshCheck.onclick = function () {
  chrome.storage.local.set({"actrefreshCheck" : ActrefreshCheck.checked}, function(){});
};