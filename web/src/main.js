import {createApp} from 'vue'
import './style.css'
import App from './App.vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import axios from 'axios'
import VueAxios from 'vue-axios'

import * as echarts from 'echarts'


const app = createApp(App)
app.use(VueAxios, axios)
app.use(ElementPlus)
app.config.globalProperties.$echarts = echarts
app.mount('#app')
