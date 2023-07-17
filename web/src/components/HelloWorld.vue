<template>
  <div class="headBar">
    <!--<img src="../assets/logo.svg" style="width: 300px;height: 20px" alt="">-->
    <el-image src="https://database-1308950705.cos.ap-chengdu.myqcloud.com/logo.svg"></el-image>
    <div class="category">
      <span style="margin-left: 50px" @click="getimg('ambulance')">Ambulance</span>
      <el-divider direction="vertical" border-style="dashed"/>
      <span @click="getimg('apple')">Apple</span>
      <el-divider direction="vertical" border-style="dashed"/>
      <span @click="getimg('bear')">Bear</span>
      <el-divider direction="vertical" border-style="dashed"/>
      <span @click="getimg('bicycle')">Bicycle</span>
      <el-divider direction="vertical" border-style="dashed"/>
      <span @click="getimg('bird')">Bird</span>
      <el-divider direction="vertical" border-style="dashed"/>
      <span @click="getimg('bus')">Bus</span>
      <el-divider direction="vertical" border-style="dashed"/>
      <span @click="getimg('cat')">Cat</span>
      <el-divider direction="vertical" border-style="dashed"/>
      <span @click="getimg('foot')">Foot</span>
      <el-divider direction="vertical" border-style="dashed"/>
      <span @click="getimg('owl')">Owl</span>
      <el-divider direction="vertical" border-style="dashed"/>
      <span @click="getimg('pig')">Pig</span>
    </div>
  </div>
  <div class="container">
    <div class="wrapper">
      <div id="content" style="float: left;">
        <div class="toolBar">
          <el-button @click="clear" style="margin-left: 5px">清空</el-button>
          <el-button @click="exprot">导出</el-button>
          <el-button @click="eraser">{{ text }}</el-button>
          <div style="margin-left: 5px">
            <!-- Element-Plus 颜色选择器 -->
            <el-color-picker v-model="color1" @change="colorChange"/>
          </div>
          <div style="margin-left: 12px;width: 100px;margin-right: 20px">
            <!-- Element-Plus 滑块 -->
            <el-slider style="margin-left: 12px;width: 100px" v-model="value1" @change="numberChange"/>
          </div>
          <div class="button-group">
            <el-upload
                action="/api/getResTest"
                method="post"
                :on-success="handleUploadSuccess"
                :on-error="handleUploadError"
                :before-upload="beforeUpload"
                :show-file-list="false"
                style="margin-top: 20px;margin-right: 10px">
              <el-button type="primary" style="margin-left: 20px">CNN识别</el-button>
            </el-upload>
            <el-upload
                action="/api/getResTestLstm"
                method="post"
                :on-success="handleUploadSuccess"
                :on-error="handleUploadError"
                :before-upload="beforeUpload2"
                :show-file-list="false"
                style="margin-top: 20px;margin-right: 10px">
              <el-button type="primary" style="margin-left: 20px">LSTM识别</el-button>
            </el-upload>
          </div>
        </div>
        <canvas id="myCanvas"></canvas>
      </div>
      <div style="margin-top: 100px;margin-left: 20px">
        <div v-for="(row, index) in imgRows" :key="index">
          <img style="width: 128px;height: 128px;margin-left: 10px;margin-bottom: 30px" v-for="(image, i) in row"
               :key="i" :src="'data:image/png;base64,' + image" alt=""/>
        </div>
      </div>
    </div>
    <el-divider content-position="center">预测结果展示</el-divider>
    <div class="result">
      <div ref="echart" class="echartDiv" id="echart" style="display: none"></div>
      <div v-if="success_handle" class="imageDiv" style="float:left;">
        <img :src="imageDataUrl" alt="上传的图片">
      </div>
    </div>
  </div>
</template>


<script setup>
import {reactive, ref, onMounted, nextTick} from 'vue'
import {useRouter} from "vue-router";
import axios from "axios";
import * as echarts from 'echarts';
//变量命名
const imageDataUrl = ref(null)
const echart = ref(null);
const text = ref('橡皮擦')
const textFlag = ref(true)
const color1 = ref('#409EFF')
const value1 = ref(6)
const success_handle = ref(false)
let printRecord = [];  // 用于记录绘画过程的数组
let myCanvas;
let ctx;
let isMouseDown;
let strokeStyle;
let lineWidth;
let PrintRecord = []
let lastX = 0;
let lastY = 0;
let penRecord = []
const result = reactive({
  ambulance: 0,
  apple: 0,
  bear: 0,
  bicycle: 0,
  bird: 0,
  bus: 0,
  cat: 0,
  foot: 0,
  owl: 0,
  pig: 0
})
const imgRows = ref();
const preimg = ref(null);

function splitImagesIntoRows(images, rowSize) {
  const rows = [];
  let currentRow = [];
  images.forEach(image => {
    currentRow.push(image);
    if (currentRow.length === rowSize) {
      rows.push(currentRow);
      currentRow = [];
    }
  });
  if (currentRow.length > 0) {
    rows.push(currentRow);
  }
  return rows;
}

function getimg(category) {
  const data = {
    class_name_2_num_dict: {
      ambulance: 0,
      apple: 0,
      bear: 0,
      bicycle: 0,
      bird: 0,
      bus: 0,
      cat: 0,
      foot: 0,
      owl: 0,
      pig: 0
    },
    img_width: 120
  };
  data.class_name_2_num_dict[category] = 20;
  // console.log(data)
  axios.post('/api/getExample', data, {
    headers: {
      'Content-Type': 'application/json'
    }
  })
      .then(response => {
        let all_images = response.data[category];
        // console.log(all_images)
        imgRows.value = splitImagesIntoRows(all_images, 5);
      })
      .catch(error => {
        console.error(error);
      });
}

function showWrapper2() {
  var wrapper2 = document.getElementById("echart");
  wrapper2.style.display = "block";
}

// printRecord.push(penRecord)
const router = useRouter();

function backIndex() {
  router.push('/')
}

function beforeUpload(file) {
  // 将文件保存到本地变量
  const reader = new FileReader()
  reader.readAsDataURL(file)
  reader.onload = (event) => {
    imageDataUrl.value = event.target.result
  }
}


const echartInit = (x, y) => {
  if (!echart.value) return;
  var myChart = echarts.init(echart.value);
  var option = {
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "shadow",
      },
      formatter: function (parms) {
        var str = parms[0].axisValue + "</br>" + parms[0].marker + "预测" + parms[0].value;
        return str;
      },
    },
    textStyle: {
      color: "#333",
    },
    color: ["#7BA9FA", "#4690FA"],
    grid: {
      containLabel: true,
      left: "10%",
      top: "20%",
      bottom: "10%",
      right: "10%",
    },
    xAxis: {
      type: "category",
      data: x,
      axisLine: {
        lineStyle: {
          color: "#333",
        },
      },
      axisTick: {
        show: false,
      },
      axisLabel: {
        margin: 20, //刻度标签与轴线之间的距离。
        textStyle: {
          color: "#000",
        },
        interval: 0,
      },
    },
    yAxis: {
      type: "value",
      axisLine: {
        show: true,
        lineStyle: {
          color: "#B5B5B5",
        },
      },
      splitLine: {
        lineStyle: {
          // 使用深浅的间隔色
          color: ["#B5B5B5"],
          type: "dashed",
          opacity: 0.5,
        },
      },
      axisLabel: {},
    },
    series: [
      {
        data: y,
        stack: "zs",
        type: "bar",
        barMaxWidth: "auto",
        barWidth: 30,
        itemStyle: {
          color: {
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            type: "linear",
            global: false,
            colorStops: [
              {
                offset: 0,
                color: "#5EA1FF",
              },
              {
                offset: 1,
                color: "#90BEFF",
              },
            ],
          },
        },
      },
      {
        data: y,
        type: "pictorialBar",
        barMaxWidth: "20",
        symbolPosition: "end",
        symbol: "diamond",
        symbolOffset: [0, "-50%"],
        symbolSize: [30, 10],
        zlevel: 2,
      },
    ],
  };
  myChart.setOption(option);
};
// myCanvas.width = window.innerWidth
// myCanvas.height = window.innerHeight
// myCanvas.addEventListener('mousedown', function (e) {
//   isMouseDown = true
//   ctx.moveTo(e.pageX, e.pageY)
//   ctx.beginPath();
//   ctx.lineWidth = lineWidth || value1.value;
//   ctx.strokeStyle = strokeStyle || color1.value;
// })
function initCanvas() {
  let dpr = window.devicePixelRatio || 1;
  myCanvas = document.getElementById('myCanvas')
  ctx = myCanvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, ctx.width, ctx.height); // 绘制填充矩形来填充整个Canvas
  let rect = myCanvas.getBoundingClientRect();
  myCanvas.width = rect.width;
  myCanvas.height = rect.height;
  myCanvas.addEventListener('mousedown', function (e) {
    // const currentX = e.pageX - myCanvas.offsetLeft;
    // const currentY = e.pageY - myCanvas.offsetTop;
    // if (printRecord.length === 0) {
    //   lastX = currentX;
    //   lastY = currentY;
    // } else {
    //   const offsetX = currentX - lastX;
    //   const offsetY = currentY - lastY;
    //   lastX = currentX;
    //   lastY = currentY;
    //   const penRecord = [offsetX, offsetY, 1];
    //   printRecord.push(penRecord);
    //   console.log(penRecord)
    // }
    if (printRecord.length !== 0) {
      const rect = myCanvas.getBoundingClientRect();
      const X = e.clientX - rect.left;
      const Y = e.clientY - rect.top;
      printRecord.push([X, Y - 52, 1])
      console.log(X, Y - 52, 1)
    }
    isMouseDown = true
    ctx.moveTo(e.pageX - myCanvas.offsetLeft, e.pageY - myCanvas.offsetTop)
    ctx.beginPath();
    ctx.lineWidth = lineWidth || value1.value;
    ctx.strokeStyle = strokeStyle || color1.value;
    // 添加开始绘画的记录
  })

  // function delayedFunction1(e) {
  //
  //   // 添加绘画中的记录
  //   const currentX = e.pageX - myCanvas.offsetLeft;
  //   const currentY = e.pageY - myCanvas.offsetTop;
  //   if (printRecord.length === 0) {
  //     lastX = currentX;
  //     lastY = currentY;
  //   }
  //   const offsetX = currentX - lastX;
  //   const offsetY = currentY - lastY;
  //   lastX = currentX;
  //   lastY = currentY;
  //   penRecord = [offsetX, offsetY, 0]
  //   printRecord.push(penRecord);
  //   console.log(penRecord)
  // }
  //
  // function delayedFunction2(e) {
  //   const currentX = e.pageX - myCanvas.offsetLeft;
  //   const currentY = e.pageY - myCanvas.offsetTop;
  //   if (printRecord.length === 0) {
  //     lastX = currentX;
  //     lastY = currentY;
  //   }
  //   const offsetX = currentX - lastX;
  //   const offsetY = currentY - lastY;
  //   lastX = currentX;
  //   lastY = currentY;
  //   const penRecord = [offsetX, offsetY, 1];
  //   printRecord.push(penRecord);
  //   console.log(penRecord)
  // }

  // setTimeout(delayedFunction, 600)
  myCanvas.addEventListener('mousemove', function (e) {
    if (isMouseDown) {
      const rect = myCanvas.getBoundingClientRect();
      const X = e.clientX - rect.left;
      const Y = e.clientY - rect.top;
      ctx.lineTo(X, Y)
      ctx.stroke()
      console.log(X, Y - 52, 0)
      // // 添加绘画中的记录
      // const currentX = e.pageX - myCanvas.offsetLeft;
      // const currentY = e.pageY - myCanvas.offsetTop;
      // if (printRecord.length === 0) {
      //   lastX = currentX;
      //   lastY = currentY;
      // }
      // const offsetX = currentX - lastX;
      // const offsetY = currentY - lastY;
      // lastX = currentX;
      // lastY = currentY;
      penRecord = [X, Y - 52, 0]
      printRecord.push(penRecord);
    }
    // else {
    //   // setTimeout(delayedFunction2(e), 1000)
    // }
    // else {
    //   const currentX = e.pageX - myCanvas.offsetLeft;
    //   const currentY = e.pageY - myCanvas.offsetTop;
    //   if (printRecord.length === 0) {
    //     lastX = currentX;
    //     lastY = currentY;
    //   }
    //   const offsetX = currentX - lastX;
    //   const offsetY = currentY - lastY;
    //   lastX = currentX;
    //   lastY = currentY;
    //   const penRecord = [offsetX, offsetY, 1];
    //   printRecord.push(penRecord);
    //   console.log(penRecord)
    // }
  })
  myCanvas.addEventListener('mouseup', function (e) {
    isMouseDown = false;
  })
}

function clear() {
  printRecord = []
  let rect = myCanvas.getBoundingClientRect();
  myCanvas.width = rect.width;
  myCanvas.height = rect.height;
}

  function beforeUpload2() {
    imageDataUrl.value = localStorage.getItem('imageDataUrl');
  }

function exprot() {
  let link = document.createElement('a')
  link.href = myCanvas.toDataURL('image/png')
  localStorage.setItem('imageDataUrl', link.href);
  link.download = 'draw.png'
  link.click()

  let paintRecordStr = '';
  // 逐行构建文本内容
  for (let i = 0; i < printRecord.length; i++) {
    const record = printRecord[i];
    const line = record.join(',') + '\n'; // 使用逗号分隔每个值，并添加换行符
    paintRecordStr += line;
  }
  // 创建Blob对象
  const txtBlob = new Blob([paintRecordStr], {type: 'text/plain;charset=utf-8'});
  // 创建一个新的a标签
  const txtLink = document.createElement('a');
  txtLink.href = URL.createObjectURL(txtBlob);
  txtLink.download = 'paint_record.txt';
  // 模拟点击下载链接
  txtLink.click();
  // 释放URL对象
  // URL.revokeObjectURL(txtLink.href);
}

function colorChange(e) {
  strokeStyle = e
}

function numberChange(e) {
  lineWidth = e
}

function eraser(e) {
  textFlag.value = !textFlag.value
  if (!textFlag.value) {
    // cursorIcon.value = "url(/src/assets/.svg),default"
    text.value = '画笔'
    strokeStyle = '#ffffff'
  } else {
    // cursorIcon.value = 'url(/src/assets/pen.svg),default'
    text.value = '橡皮擦'
    colorChange(e)
  }
}

function handledata(response) {
  showWrapper2()
  result.ambulance = response.predict_class.ambulance.toFixed(3);
  result.apple = response.predict_class.apple.toFixed(3);
  result.bear = response.predict_class.bear.toFixed(3);
  result.bicycle = response.predict_class.bicycle.toFixed(3);
  result.bird = response.predict_class.bird.toFixed(3);
  result.bus = response.predict_class.bus.toFixed(3);
  result.cat = response.predict_class.cat.toFixed(3);
  result.foot = response.predict_class.foot.toFixed(3);
  result.owl = response.predict_class.owl.toFixed(3);
  result.pig = response.predict_class.pig.toFixed(3);
  const xAxisData = ['ambulance', 'apple', 'bear', 'bicycle', 'bird', 'bus', 'cat', 'foot', 'owl', 'pig'];
  const yAxisData = [result.ambulance, result.apple, result.bear, result.bicycle
    , result.bird, result.bus, result.cat, result.foot, result.owl, result.pig];
  success_handle.value = true
  echartInit(xAxisData, yAxisData)
}
// function handledata2(response) {
//   showWrapper2()
//   result.ambulance = response.predict_class.ambulance.toFixed(3);
//   result.apple = response.predict_class.apple.toFixed(3);
//   result.bear = response.predict_class.bear.toFixed(3);
//   result.bicycle = response.predict_class.bicycle.toFixed(3);
//   result.bird = response.predict_class.bird.toFixed(3);
//   result.bus = response.predict_class.bus.toFixed(3);
//   result.cat = response.predict_class.cat.toFixed(3);
//   result.foot = response.predict_class.foot.toFixed(3);
//   result.owl = response.predict_class.owl.toFixed(3);
//   result.pig = response.predict_class.pig.toFixed(3);
//   const xAxisData = ['ambulance', 'apple', 'bear', 'bicycle', 'bird', 'bus', 'cat', 'foot', 'owl', 'pig'];
//   const yAxisData = [result.ambulance, result.apple, result.bear, result.bicycle
//     , result.bird, result.bus, result.cat, result.foot, result.owl, result.pig];
//   success_handle.value = true
//   echartInit(xAxisData, yAxisData)
// }
function handleUploadSuccess(response) {
  handledata(response)
  console.log(response)
}

function handleUploadSuccess2(response) {
  handledata(response)

}

function lstm() {
  // 发送POST请求给后端

  axios.post('/api/', printRecord, {
    headers: {
      'Content-Type': 'application/json',
    },
  })
      .then(response => {
        success_handle.value = true
        imageDataUrl.value = myCanvas.toDataURL(); // 获取Canvas的数据URL
      })
      .catch(error => {
        console.error('Failed to send data to backend:', error);
      });
}

onMounted(() => {
  nextTick(() => {
    initCanvas()
    getimg('ambulance')
  })
})
</script>


<style scoped>
html,
.echartDiv {
  grid-column: 1 / 2;
  width: 100%;
  height: 400px;
}

.result {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-gap: 10px;
}

.imageDiv {
  grid-column: 2 / 3;
  display: flex;
  justify-content: center;
  align-items: center;
}

.imageDiv img {
  max-width: 100%;
  max-height: 400px;
}

body {
  width: 100%;
  height: 100%;
  margin: 0;
  box-sizing: border-box;
  overflow-y: hidden;
  overflow-x: hidden;
}

.container {
  display: grid;
  grid-template-rows: 1fr 1fr; /* 设置两行，每行比例为1：1 */
  height: 100vh; /* 设置容器高度为视口高度 */
}

.button-group {
  margin-left: 10px;
  justify-content: space-between;
  display: flex;
  align-items: center;
  margin-bottom: 20px;
}

.button-container {
  display: flex;
  justify-content: center;
  align-items: center;
}

span {
  margin-right: 30px;
}

.category {
  /*font-size: 18px;*/
  font: 700 26px 华文彩云;
}

#content {
  margin-top: 60px;
  width: 700px;
  height: 700px;
  position: relative;
}

.headBar {
  width: 100%;
  height: 50px;
  background-color: #ffd139;
  box-shadow: 0 5px 2px #e8e8e8;
  position: absolute;
  top: 0;
  left: 0;
  display: flex;
  align-items: center;
  box-sizing: border-box;
}

.toolBar {
  width: 100%;
  height: 50px;
  background-color: #ffd139;
  box-shadow: 0 5px 2px #e8e8e8;
  position: absolute;
  top: 0;
  left: 0;
  display: flex;
  align-items: center;
  box-sizing: border-box;
}

#myCanvas {
  /*margin-top: 300px;*/
  width: 700px;
  height: 700px;
  /*display: block;*/
  border: 1px solid #ccc;
  cursor: v-bind(cursorIcon);
  overflow-y: hidden;
  overflow-x: hidden;
}

.light {
  width: 40px;
  height: 30px;
  position: absolute;
  top: 30px;
  right: 30px;
  transform: translate(-50%, -50%);
  text-align: center;
  line-height: 30px;
  color: #03e9f4;
  font-size: 20px;
  text-transform: uppercase;
  transition: 0.5s;
  letter-spacing: 4px;
  cursor: pointer;
  overflow: hidden;
}

.light:hover {
  background-color: #03e9f4;
  color: #050801;
  box-shadow: 0 0 5px #03e9f4,
  0 0 25px #03e9f4,
  0 0 50px #03e9f4,
  0 0 200px #03e9f4;
}

.light div {
  position: absolute;
}

.light div:nth-child(1) {
  width: 100%;
  height: 2px;
  top: 0;
  left: -100%;
  background: linear-gradient(to right, transparent, #03e9f4);
  animation: animate1 1s linear infinite;
}

.light div:nth-child(2) {
  width: 2px;
  height: 100%;
  top: -100%;
  right: 0;
  background: linear-gradient(to bottom, transparent, #03e9f4);
  animation: animate2 1s linear infinite;
  animation-delay: 0.25s;
}

.light div:nth-child(3) {
  width: 100%;
  height: 2px;
  bottom: 0;
  right: -100%;
  background: linear-gradient(to left, transparent, #03e9f4);
  animation: animate3 1s linear infinite;
  animation-delay: 0.5s;
}

.light div:nth-child(4) {
  width: 2px;
  height: 100%;
  bottom: -100%;
  left: 0;
  background: linear-gradient(to top, transparent, #03e9f4);
  animation: animate4 1s linear infinite;
  animation-delay: 0.75s;
}

@keyframes animate1 {
  0% {
    left: -100%;
  }

  50%,
  100% {
    left: 100%;
  }
}

@keyframes animate2 {
  0% {
    top: -100%;
  }

  50%,
  100% {
    top: 100%;
  }
}

@keyframes animate3 {
  0% {
    right: -100%;
  }

  50%,
  100% {
    right: 100%;
  }
}

@keyframes animate4 {
  0% {
    bottom: -100%;
  }
  50%,
  100% {
    bottom: 100%;
  }
}


</style>
