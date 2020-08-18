function getFuelValue() {
  var uiFuel = document.getElementsByName("uiFuel");
  for(var i in uiFuel) {
    if(uiFuel[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function getTransValue() {
  var uiTrans = document.getElementsByName("uiTrans");
  for(var i in uiTrans) {
    if(uiTrans[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function getOwnerValue() {
  var uiOwner = document.getElementsByName("uiOwner");
  for(var i in uiOwner) {
    if(uiOwner[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function onClickedEstimatePrice() {
  console.log("Estimate price button clicked");
  var Year = document.getElementById("uiYear");
  var Mileage = document.getElementById("uiMileage");
  var Engine = document.getElementById("uiEngine");
  var Power = document.getElementById("uiPower");
  var Fuel = getFuelValue();
  var Trans = getTransValue();
  var Owner = getOwnerValue();
  var Company = document.getElementById("uiCompany");
  var Model = document.getElementById("uiModel");
  var estPrice = document.getElementById("uiEstimatedPrice");

   var url = "http://127.0.0.1:5000/predict_car_price"; // Use this if you are NOT using nginx which is first 7 tutorials
  //var url = "/api/predict_home_price"; // Use this if  you are using nginx. i.e tutorial 8 and onwards

  $.post(url, {
      Year: parseInt(Year.value),
      Engine: parseInt(Engine.value),
      Mileage: parseFloat(Mileage.value),
      Power: parseFloat(Power.value),
      Fuel_Type: Fuel,
      Transmission: Trans,
      Owner_Type: Owner,
      Company: Company.value,
      Model: Model.value
  },function(data, status) {
      console.log(data.estimated_price);
      estPrice.innerHTML = "<h2>" + data.estimated_price.toString() + " Lakh</h2>";
      console.log(status);
  });
}

function onPageLoad() {
  console.log( "document loaded" );
   var url = "http://127.0.0.1:5000/get_company_names"; // Use this if you are NOT using nginx which is first 7 tutorials
  //var url = "/api/get_location_names"; // Use this if  you are using nginx. i.e tutorial 8 and onwards
  $.get(url,function(data, status) {
      console.log("got response for get_company_names request");
      if(data) {
          var Company = data.company;
          var uiCompany = document.getElementById("uiCompany");
          $('#uiCompany').empty();
          for(var i in Company) {
              var opt = new Option(Company[i]);
              $('#uiCompany').append(opt);
          }
      }
  });
  
  var url = "http://127.0.0.1:5000/get_models_names"; // Use this if you are NOT using nginx which is first 7 tutorials
  //var url = "/api/get_location_names"; // Use this if  you are using nginx. i.e tutorial 8 and onwards
  $.get(url,function(data, status) {
      console.log("got response for get_models_names request");
      if(data) {
          var Model = data.models;
          var uiModel = document.getElementById("uiModel");
          $('#uiModel').empty();
          for(var i in Model) {
              var opt = new Option(Model[i]);
              $('#uiModel').append(opt);
          }
      }
  });
}

window.onload = onPageLoad;
