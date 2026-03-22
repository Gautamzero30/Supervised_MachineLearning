const express = require("express");
let app = express();[]
const port = 8080;
const path = require("path")
app.use(express.static(path.join(__dirname,"/public/js")));
app.use(express.static(path.join(__dirname,"/public/css")));


app.set("view engine","ejs");
app.set("views",path.join(__dirname,"/views"));
app.get("/",(req,res)=>{
    res.render("home.ejs");

})
app.get("/hello",(req,res)=>{
    res.send("<h1>hello </h1>");

}) 
app.get("/rolldice",(req,res)=>{
    let diceval = Math.floor(Math.random()*6+1)
    res.render("rolldice.ejs",{diceval});

}) 
// app.get("/ig/:username",(req,res)=>
// {    const followers = ["adam","smith","marshal","karl","marx","einstein"]
//     let {username} = req.params
//     console.log(username); 
//     res.render("instagram.ejs",{username,followers})
// })
app.get("/ig/:username",(req,res)=>{
    let {username}= req.params;
    const instaData=require("./data.json")
    const data = instaData[username]
    
    if(data){
        res.render("instagram.ejs",{data})

    }
    else{
        res.render("error.ejs")

    }
    console.log(data)
    console.log(instaData)


})

app.listen((port),()=>
{   console.log(`listening on port ${port}`)

})
