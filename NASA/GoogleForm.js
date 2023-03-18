
var ssID = "1giE7fNubfOgWnR6UgdjKa3W-5qpYJx60bfk-vu120fI" // ID of Google Sheet with list of file names in it.

// var ssID = "11BQo8Nnjq6l34WeuViWW5afNXG9DZuXthjOw87CBvbo"; //1r6rLmbqt-M5Z3AZFsj1ci1ZReTLj3QFX5DPW_htS6L4
// var formID = "1Xv76C8CNU_hWPp0TxjCTmGOLl8l-WyzCKqxybnS7iqE";

var formID = "1BoNi_lA9I3TCmNbx--_jYoKNyHX-7PylCpi3J4ZVcAQ"; // ID of Google Form
var form = FormApp.openById(formID);
var wsdata = SpreadsheetApp.openById(ssID).getSheetByName("set_2_form_2").getDataRange().getValues();


function myFunction() {

  var image1_name = "";

  var image2_name = "";

  var files = "";

  var file = "";

  var image_split_arr = "";

  var img = "";

 

  wsdata.forEach(function (data, i) {
    files = DriveApp.getFilesByName(data[1]); // data[1] is column name NDVI_TS_Name. it starts indexing at zero

    if (files.hasNext()) {

      file = files.next();

      image1_name = file.getUrl();

      // Logger.log(i+"Img1"+image1_name);

      image_split_arr = image1_name.split("/");

      img = DriveApp.getFileById(image_split_arr[5]);

      form.addImageItem()

        .setImage(img);
    }

    // var form = FormApp.openById('1234567890abcdefghijklmnopqrstuvwxyz');


// http://www.google.com/maps/place/"+data[6]+","+data[7]+"/@"+data[6]+","+data[7]+",14z/data=!3m1!1e3")

      // http://www.google.com/maps/place/46.8327393,-118.9120915/@46.8327393,-118.9120915,14z/data=!3m1!1e3

      // https://www.google.com/maps/@"+data[6]+","+data[7]+",14z/data=!3m1!1e3")

    // Logger.log(data)

    if (i > 0) {

      // var item1 = form.addMultipleChoiceItem();

      // item1.setTitle("QUESTION " + data[0] + " : How would you classify? \n http://www.google.com/maps/place/"+data[6]+","+data[7]+"/@"+data[6]+","+data[7]+",16z/data=!3m1!1e3")

      //   .setChoices([

      //     item1.createChoice('Single Crop'),

      //     item1.createChoice('Double Crop/Cover crop'),

      //     item1.createChoice('Unsure')

      //   ]);

var item = form.addGridItem();

item.setTitle("QUESTION " + i + " : How do you classify? \n Ignore late-fall/winter signatures such as fall planting, winter cover crops, or volunteers/weeds. \n http://www.google.com/maps/place/"+ data[4]+ "," + data[5] + "/@"+data[4] + "," + data[5] + ",16z/data=!3m1!1e3" + " (" + data[0] + ")")

    .setRows(['Choose one: '])

    .setColumns(['Single Crop', 'Double Crop', 'Mustard Crop', 'Either double or mustard crop', 'Unsure']).setRequired(true);

 

      //   var item2 = form.addGridItem;

      //   // var choices = item.asGridItem();

      // item2.setTitle("QUESTION: If double/cover crop, can you distinguish?")

      // // choices.setColumns(answers);

      // //   choices.setRows(questions);

      //   .setChoices([

      //     item2.createChoice('Unsure' ),

      //     item2.createChoice('Double Crop'),

      //     item2.createChoice('Cover Crop'),

      //   ]);

var item2 = form.addCheckboxItem();

item2.setTitle('Discuss as a full group?')

    .setChoices([

          item2.createChoice('Check for Yes')

    ]);
        var comments = form.addParagraphTextItem();

        comments.setTitle('Comments');

    }

    form.addPageBreakItem()

  });

}