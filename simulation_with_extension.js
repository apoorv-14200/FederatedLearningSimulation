//Simulation for resource efficient federated learning

//let there be m types of resources

let m = 5;

let edge_nodes_data = [
  [[5376, 5118, 5239, 5031, 5367], 158, 0],
  [[5107, 5440, 5247, 5238, 5201], 210, 0],
  [[5414, 5072, 5273, 5156, 5264], 371, 0],
  [[5463, 5090, 5090, 5034, 5374], 260, 0],
  [[5407, 5423, 5499, 5364, 5337], 382, 1],
  [[5115, 5219, 5237, 5441, 5476], 301, 1],
  [[5035, 5368, 5217, 5021, 5227], 81, 1],
  [[5484, 5035, 5257, 5485, 5031], 243, 1],
  [[5385, 5391, 5201, 5161, 5323], 317, 2],
  [[5270, 5217, 5018, 5308, 5424], 224, 2],
  [[5459, 5097, 5204, 5225, 5459], 428, 2],
  [[5214, 5500, 5404, 5007, 5259], 306, 2],
];

//out of these 5 resources
//let first and second of these resources denote the resources which are reponsible for computation

//third resource is for bandwidth

//we define resource limits for each resources
let CLU_flag = false;

class ParameterServer {
  constructor(threshold_for_updates, recluster_threshold, model) {
    this.threshold_for_updates = threshold_for_updates;
    this.extent_of_updates = 0;
    // this.resources_lim = resources_lim;
    this.aggregation_queue = [];
    this.recluster_threshold = recluster_threshold;
    this.tb = 0;
    this.model = model;
  }
  StartglobalAggregation() {
    let start = new Date().getTime();
    let serverloop = setInterval(() => {
      if (this.aggregation_queue.length > 0) {
        let cur_update = this.aggregation_queue.shift();
        this.extent_of_updates += cur_update[0];
        console.log("Extent_of_update", this.extent_of_updates);
        this.tb += 1;
        if (this.extent_of_updates > this.threshold_for_updates) {
          let end = new Date().getTime();
          console.log("Total Training finished in time ms", end - start);
          clearInterval(serverloop);
          process.exit();
        }
        //adaptive reclustering
        if (this.tb == this.recluster_threshold) {
          this.tb = 0;
          for (i = 0; i < edge_nodes.length; i++) {
            edge_nodes[i].training = false;
          }
          setTimeout(() => {
            this.Reclustering();
          }, 20);
        } else {
          edge_nodes[cur_update[1]].get_model_from_PS(this.model);
        }
      }
    }, 1);
  }
  StartServer() {
    console.log(
      "Server is Up for receiving models and dispatched the model to leader nodes"
    );
    let i = 0;
    this.StartglobalAggregation();
    let SendNodes = setInterval(() => {
      if (i == leaderNodes.length) {
        clearInterval(SendNodes);
      }
      if (i < leaderNodes.length) {
        edge_nodes[leaderNodes[i]].get_model_from_PS(this.model);
      }
      i += 1;
    }, 5);
  }
  Reclustering() {
    console.log("reclustering started");
    let currentIndex = edge_nodes.length,
      randomIndex;
    // While there remain elements to shuffle...
    while (currentIndex != 0) {
      // Pick a remaining element...
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;

      // And swap it with the current element.
      [edge_nodes[currentIndex], edge_nodes[randomIndex]] = [
        edge_nodes[randomIndex],
        edge_nodes[currentIndex],
      ];
    }
    let i;
    let idx = 0;
    for (i = 0; i < num_of_clusters; i++) {
      let p = num_nodes_in_one_cluster;
      let mxid = -1;
      let mx = -1;
      while (p--) {
        //selecting node with max bacndwidth as leader node
        edge_nodes[idx].clusterid = i;
        if (edge_nodes[idx].resources[3] > mx) {
          mx = edge_nodes[idx].resources[3];
          mxid = idx;
        }
        leaderNodes[i] = mxid;
        idx++;
      }
    }
    console.log("Reclustering Done");
    for (i = 0; i < leaderNodes.length; i++) {
      edge_nodes[leaderNodes[i]].get_model_from_PS(this.model);
    }
    return;
  }
}

class Model {
  constructor(num_of_layers, nodes_in_layers, model_size) {
    //num of layers in our model
    this.model_size = model_size;
    this.num_of_layers = num_of_layers;
    //num_of nodes in each in layer
    this.nodes_in_layers = nodes_in_layers;
    let i = this.num_of_layers;
    let num_of_operations = 1;
    for (i = 0; i < this.nodes_in_layers.length; i++) {
      num_of_operations = num_of_operations * this.nodes_in_layers[i];
    }
    this.num_of_operations = num_of_operations;
  }
  get_operations() {
    return this.num_of_operations;
  }
}

class edgeNode {
  constructor(
    resources,
    data,
    clusterid,
    num_of_local_epoch,
    node_num,
    server
  ) {
    this.resources = resources;
    this.num_of_datapoints = data;
    this.node_num = node_num;
    this.clusterid = clusterid;
    this.num_of_local_epoch = num_of_local_epoch;
    this.num_of_epochs_after_prev_aggregation = 0;
    this.training = false;
    this.server = server;
    //this is used if the current node becomes leadernode of its cluster
    this.model_queue = [];
    this.StoredModel = 0;
    this.aggregationInterval;
  }
  getdata() {
    return this.privatedata;
  }
  train_model(model) {
    console.log(`${this.node_num} Received model and started training`);
    let computational_resource = (this.resources[0] + this.resources[1]) / 2;
    let time_to_train =
      (model.get_operations() * this.num_of_datapoints) /
      computational_resource;
    let time_to_send = model.model_size / this.resources[2];
    let train_start = new Date().getTime();
    this.training = true;
    let train = setInterval(() => {
      let cur = new Date().getTime();
      if (
        cur - train_start >= time_to_train + time_to_send ||
        this.training == false
      ) {
        let extent_of_completion = 1;
        if (cur - train_start < time_to_train + time_to_send) {
          extent_of_completion =
            (cur - train_start) / (time_to_train + time_to_send);
        }
        console.log(
          `Total Time to train node ${this.node_num}`,
          time_to_train + time_to_send
        );
        this.training = false;
        clearInterval(train);
        //to be changed
        edge_nodes[leaderNodes[this.clusterid]].get_model_from_cluster_nodes(
          extent_of_completion * this.num_of_datapoints
        );
      }
    }, 1);
  }
  get_model_from_PS(model) {
    let i;
    console.log(
      `LN ${this.node_num} got model from PS and broadcasted to its cluster nodes`
    );
    for (i = 0; i < edge_nodes.length; i++) {
      if (
        edge_nodes[i].clusterid == this.clusterid &&
        edge_nodes[i].node_num != this.node_num
      ) {
        edge_nodes[i].train_model(model);
      }
    }
  }
  get_model_from_cluster_nodes(extent_of_completion) {
    this.model_queue.push(extent_of_completion);
    if (this.model_queue.length == num_nodes_in_one_cluster - 1) {
      console.log(
        `LN received model from all cluster nodes and performing aggreagation`
      );
      this.LeaderNodeaggregation();
    }
  }
  LeaderNodeaggregation() {
    let sum = 0;
    for (i = 0; i < this.model_queue.length; i++) {
      sum += this.model_queue[i];
    }
    let avgimportance = sum / this.model_queue.length;
    console.log(
      `LeaderNode ${this.node_num} Aggregated with importance ${avgimportance}`
    );
    this.model_queue.length = 0;
    console.log(`PS got update from LN  ${this.node_num}`);
    this.server.aggregation_queue.push([
      avgimportance,
      leaderNodes[this.clusterid],
    ]);
  }
}

function randomInteger(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function get_random_resources() {
  let r = [];
  let i = 0;
  for (i = 0; i < m; i++) {
    let num = randomInteger(5000, 5500);
    r.push(num);
  }
  return r;
}

let num_of_edge_nodes = 12;
let edge_nodes = [];
let num_of_clusters = 3;

let num_nodes_in_one_cluster = num_of_edge_nodes / num_of_clusters;

let model = new Model(5, [5, 8, 8, 8, 2], 500);
let PS = new ParameterServer(5000, 5, model);

let leaderNodes = [];
leaderNodes.length = num_of_clusters;

let i = 0;
let idx = 0;
for (i = 0; i < num_of_clusters; i++) {
  let p = num_nodes_in_one_cluster;
  let mxid = -1;
  let mx = -1;
  while (p--) {
    // let r = get_random_resources();
    // let data = randomInteger(50, 500);
    let node = new edgeNode(
      edge_nodes_data[idx][0],
      edge_nodes_data[idx][1],
      i,
      1,
      idx,
      PS
    );
    edge_nodes.push(node);
    if (node.resources[3] > mx) {
      mx = edge_nodes[idx].resources[3];
      mxid = idx;
    }
    idx++;
  }
  leaderNodes[i] = mxid;
}

for (i = 0; i < edge_nodes.length; i++) {
  console.log(
    edge_nodes[i].resources,
    edge_nodes[i].num_of_datapoints,
    edge_nodes[i].clusterid
  );
}

PS.StartServer();
