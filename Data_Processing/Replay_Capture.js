const http = require('axios').default;
const LZString = require('lz-string');
const fs=require("fs").promises;

function deserialize(serialized) {
    const obj =
        JSON.parse(LZString.decompressFromUint8Array(new Uint8Array(serialized)));
  
    if (!obj) return;
  
    const replay = {};
    let i = 0;
    replay.version = obj[i++];
    replay.id = obj[i++];
    replay.mapWidth = obj[i++];
    replay.mapHeight = obj[i++];
    replay.usernames = obj[i++];
    replay.stars = obj[i++];
    replay.cities = obj[i++];
    replay.cityArmies = obj[i++];
    replay.generals = obj[i++];
    replay.mountains = obj[i++] || [];
    replay.moves = obj[i++].map(deserializeMove);
    replay.afks = obj[i++].map(deserializeAFK);
    replay.teams = obj[i++];
    replay.map = obj[i++];
    replay.neutrals = obj[i++] || [];
    replay.neutralArmies = obj[i++] || [];
    replay.swamps = obj[i++] || [];
    replay.chat = (obj[i++] || []).map(deserializeChat);
    replay.playerColors = obj[i++] || replay.usernames.map((u, i) => i);
    replay.lights = obj[i++] || [];
  
    const options = (obj[i++] || [
      1, Constants.DEFAULT_CITY_DENSITY_OPTION, Constants.DEFAULT_MOUNTAIN_DENSITY_OPTION, Constants.DEFAULT_SWAMP_DENSITY_OPTION, // added in 11
      Constants.DEFAULT_CITY_FAIRNESS_OPTION, Constants.DEFAULT_SPAWN_FAIRNESS_OPTION, Constants.DEFAULT_DESERT_DENSITY_OPTION, Constants.DEFAULT_LOOKOUT_DENSITY_OPTION, Constants.DEFAULT_OBSERVATORY_DENSITY_OPTION // added in 13
    ]);
    replay.speed = options[0];
    replay.city_density = options[1];
    replay.mountain_density = options[2];
    replay.swamp_density = options[3];
    replay.modifiers = obj[i++] || [];
    
    // v13
    replay.observatories = obj[i++] || [];
    replay.lookouts = obj[i++] || [];
    replay.deserts = obj[i++] || [];
  
    if (replay.version > 12) {
      replay.city_fairness = options[4];
      replay.spawn_fairness = options[5];
      replay.desert_density = options[6];
      replay.lookout_density = options[7];
      replay.observatory_density = options[8];
    } else {
      replay.city_fairness = -1;
      replay.spawn_fairness = -1;
      replay.desert_density = 0.0;
      replay.lookout_density = 0.0;
      replay.observatory_density = 0.0;
    }
  
    return replay;
  };
  
  function deserializeMove(serialized) {
    return {
      index: serialized[0],
      start: serialized[1],
      end: serialized[2],
      is50: serialized[3],
      turn: serialized[4],
    };
  }
  
  function deserializeAFK(serialized) {
    return {
      index: serialized[0],
      turn: serialized[1],
    };
  }
  
  function deserializeChat(serialized) {
    return {
      text: serialized[0],
      prefix: serialized[1],
      playerIndex: serialized[2],
      turn: serialized[3],
    };
  }
  
function getReplay(replayId, server = 'na') {
    const BASE_URL = `https://generalsio-replays-${server}.s3.amazonaws.com`;
    return http.get(`${BASE_URL}/${replayId}.gior`, {responseType: 'arraybuffer'})
        .then(response => deserialize(response.data));
}
async function getData(player, name){
    var data=await getReplay(name);
    output_file = "/Users/yuxuan/Documents/AlphaGen/Replays/Game_Data/"+player+'/'+name+'.json';
    await fs.writeFile(output_file,JSON.stringify(data,null,4));
}
// test = "Gts0EzetP";
// getData("TRY", test);
Name = ["MeltedToast","Mithraaaa"];

Name.forEach(async item => {
    let input_file = `./Replays/Total_Data/filtered_${item}.json`;
    fs.mkdir('./Replays/Game_Data/'+item, { recursive: true }, (err) => {
        if (err) {
            return console.error("Error creating directory:", err);
        }
        console.log('Directory created successfully!');
    });
    try {
        const fileContent = await fs.readFile(input_file, 'utf8');
        const jsonData = JSON.parse(fileContent);
        for (const entry of jsonData) {
            // console.log(entry);
            await getData(item, entry.id);
        }
    } catch (error) {
        console.error(`Error processing ${input_file}:`, error);
    }
});
