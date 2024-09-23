
const axios=require("axios");
const fs=require("fs").promises;
async function saveReplays(username,total,path){
    var data=[];
    var cnt=200;
    for(var i=0;i<Math.ceil(total/cnt);i++){
        var offset=i*cnt;
        var response=await axios.get(`https://generals.io/api/replaysForUsername?u=${username}&offset=${offset}&count=${cnt}`)
        var replays=response.data;
        console.log(`Request ${i+1}: Retrieved ${replays.length} replays`);
        data=data.concat(replays);
    }

    await fs.writeFile(path,JSON.stringify(data,null,4));
}

const Name = ['Mithraaaa','MeltedToast']
Name.forEach(name => {
    const filePath = `./Replay/Total_Data/${name}.json`;
    saveReplays(name, 10000, filePath);
});