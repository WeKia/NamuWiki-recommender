const clampNumber = (num, a, b) =>
  Math.max(Math.min(num, Math.max(a, b)), Math.min(a, b));

$( document ).ready(function() {
    var RecentInput = $("#RecentInput");
    var Recentslider = $("#RecentSlider");

    var recent_val = 0;

    chrome.storage.local.get("recentVal", ({ recentVal }) => {
        if(recentVal) {
            recent_val = recentVal;

            Recentslider.val(recent_val);
            RecentInput.val(recent_val);
        }
        else {
            recent_val = 10;

            Recentslider.val(recent_val);
            RecentInput.val(recent_val);
        }
    });

    Recentslider.on("input", function() {
        RecentInput.val($(this).val());

        recent_val = $(this).val();

        chrome.storage.local.set({recentVal : recent_val});
    });

    RecentInput.on("change", function(){
        RecentInput.val(clampNumber(RecentInput.val(), 1, 20));
        Recentslider.val($(this).val());

        recent_val = $(this).val();

        chrome.storage.local.set({recentVal : recent_val});
    });

    var BatchInput = $("#BatchInput");
    var Batchslider = $("#BatchSlider");

    var batch_val = 0;

    chrome.storage.local.get("batchVal", ({ batchVal }) => {
        if(batchVal) {
            batch_val = batchVal;

            Batchslider.val(batch_val);
            BatchInput.val(batch_val);
        }
        else {
            batch_val = 64;

            Batchslider.val(batch_val);
            BatchInput.val(batch_val);
        }
    });

    Batchslider.on("input", function() {
        BatchInput.val($(this).val());

        batch_val = $(this).val();

        chrome.storage.local.set({batchVal : batch_val});
    });

    BatchInput.on("change", function(){
        BatchInput.val(clampNumber(BatchInput.val(), 1, 256));
        Batchslider.val($(this).val());

        batch_val = $(this).val();

        chrome.storage.local.set({batchVal : batch_val});
    });
});
