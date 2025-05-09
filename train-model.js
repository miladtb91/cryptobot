// train-model.js - فایل آموزش مدل هوش مصنوعی برای ربات تحلیل‌گر ارز دیجیتال

const tf = require('@tensorflow/tfjs-node');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const technicalIndicators = require('technicalindicators');

// تنظیمات کلی
const MODELS_DIR = path.join(__dirname, 'models');
const BINANCE_API = 'https://api.binance.com/api/v3';

// لیست ارزهای دیجیتال
const CRYPTOS = [
  { symbol: 'BTCUSDT', name: 'بیت‌کوین' },
  { symbol: 'ETHUSDT', name: 'اتریوم' },
  { symbol: 'BNBUSDT', name: 'بایننس کوین' },
  { symbol: 'SOLUSDT', name: 'سولانا' },
  { symbol: 'ADAUSDT', name: 'کاردانو' },
  { symbol: 'XRPUSDT', name: 'ریپل' },
  { symbol: 'DOGEUSDT', name: 'دوج‌کوین' },
  { symbol: 'DOTUSDT', name: 'پولکادات' },
  { symbol: 'AVAXUSDT', name: 'اولانچ' },
  { symbol: 'MATICUSDT', name: 'پلیگان' }
];

// تنظیمات آموزش مدل
const TRAINING_CONFIG = {
  epochs: 100,
  batchSize: 32,
  learningRate: 0.001,
  validationSplit: 0.2,
  timeSteps: 30,  // تعداد کندل‌های استفاده شده برای پیش‌بینی
  featureCount: 13,  // تعداد ویژگی‌های استخراج شده از هر کندل
  predictionHorizons: [1, 3, 6, 12, 24]  // افق‌های پیش‌بینی (تعداد کندل‌های آینده)
};

// ایجاد پوشه مدل‌ها اگر وجود ندارد
if (!fs.existsSync(MODELS_DIR)) {
  fs.mkdirSync(MODELS_DIR, { recursive: true });
  console.log(`پوشه "${MODELS_DIR}" ایجاد شد.`);
}

// کلاس اصلی آموزش مدل
class ModelTrainer {
  constructor() {
    this.models = {};
  }

  // دریافت داده‌های تاریخی از بایننس
  async fetchHistoricalData(symbol, interval = '1h', limit = 1000) {
    try {
      console.log(`در حال دریافت داده‌های تاریخی برای ${symbol}...`);
      
      const response = await axios.get(`${BINANCE_API}/klines`, {
        params: {
          symbol: symbol,
          interval: interval,
          limit: limit
        }
      });

      // تبدیل داده‌های خام به فرمت مناسب
      const formattedData = response.data.map(candle => ({
        timestamp: candle[0],
        open: parseFloat(candle[1]),
        high: parseFloat(candle[2]),
        low: parseFloat(candle[3]),
        close: parseFloat(candle[4]),
        volume: parseFloat(candle[5]),
        closeTime: candle[6],
        quoteAssetVolume: parseFloat(candle[7]),
        trades: parseInt(candle[8]),
        takerBuyBaseAssetVolume: parseFloat(candle[9]),
        takerBuyQuoteAssetVolume: parseFloat(candle[10])
      }));

      console.log(`${formattedData.length} کندل برای ${symbol} دریافت شد.`);
      return formattedData;
    } catch (error) {
      console.error(`خطا در دریافت داده‌های تاریخی برای ${symbol}:`, error.message);
      throw error;
    }
  }

  // استخراج ویژگی‌های تکنیکال از داده‌های خام
  extractFeatures(data) {
    if (data.length < 100) {
      throw new Error('داده‌های کافی برای استخراج ویژگی‌ها وجود ندارد.');
    }

    console.log('در حال استخراج ویژگی‌های تکنیکال...');

    // آرایه‌های مورد نیاز برای محاسبه شاخص‌ها
    const closes = data.map(d => d.close);
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const volumes = data.map(d => d.volume);
    const opens = data.map(d => d.open);

    // محاسبه شاخص‌های تکنیکال
    const sma9 = technicalIndicators.SMA.calculate({ period: 9, values: closes });
    const sma20 = technicalIndicators.SMA.calculate({ period: 20, values: closes });
    const sma50 = technicalIndicators.SMA.calculate({ period: 50, values: closes });
    const sma200 = technicalIndicators.SMA.calculate({ period: 200, values: closes });
    
    const ema9 = technicalIndicators.EMA.calculate({ period: 9, values: closes });
    const ema20 = technicalIndicators.EMA.calculate({ period: 20, values: closes });
    
    const rsi14 = technicalIndicators.RSI.calculate({ period: 14, values: closes });
    
    const macd = technicalIndicators.MACD.calculate({
      fastPeriod: 12, 
      slowPeriod: 26, 
      signalPeriod: 9, 
      values: closes
    });
    
    const bbands = technicalIndicators.BollingerBands.calculate({
      period: 20,
      stdDev: 2,
      values: closes
    });
    
    const atr = technicalIndicators.ATR.calculate({
      high: highs,
      low: lows,
      close: closes,
      period: 14
    });
    
    const stochastic = technicalIndicators.Stochastic.calculate({
      high: highs,
      low: lows,
      close: closes,
      period: 14,
      signalPeriod: 3
    });
    
    // محاسبه تغییرات قیمت
    const priceChanges = [];
    for (let i = 1; i < closes.length; i++) {
      priceChanges.push((closes[i] - closes[i-1]) / closes[i-1]);
    }
    priceChanges.unshift(0); // اضافه کردن یک مقدار صفر در ابتدا برای حفظ طول آرایه
    
    // محاسبه نسبت حجم
    const volumeRatio = [];
    for (let i = 1; i < volumes.length; i++) {
      volumeRatio.push(volumes[i] / volumes[i-1]);
    }
    volumeRatio.unshift(1); // اضافه کردن یک مقدار 1 در ابتدا برای حفظ طول آرایه

    // محاسبه محدوده روزانه (بالاترین - پایین‌ترین)
    const dailyRange = data.map(d => (d.high - d.low) / d.low);
    
    // ترکیب همه ویژگی‌ها
    const features = [];
    const labels = [];
    
    // پر کردن آرایه‌ها با مقادیر پیش‌فرض تا همه طول یکسانی داشته باشند
    const fillMissingValues = (arr, length, defaultValue = 0) => {
      while (arr.length < length) {
        arr.unshift(defaultValue);
      }
      return arr;
    };
    
    const dataLength = data.length;
    const filledSMA9 = fillMissingValues(sma9, dataLength);
    const filledSMA20 = fillMissingValues(sma20, dataLength);
    const filledSMA50 = fillMissingValues(sma50, dataLength);
    const filledSMA200 = fillMissingValues(sma200, dataLength);
    const filledEMA9 = fillMissingValues(ema9, dataLength);
    const filledEMA20 = fillMissingValues(ema20, dataLength);
    const filledRSI = fillMissingValues(rsi14, dataLength, 50);
    
    // پر کردن MACD
    const filledMACD = Array(dataLength).fill({ MACD: 0, signal: 0, histogram: 0 });
    for (let i = 0; i < macd.length; i++) {
      filledMACD[i + (dataLength - macd.length)] = macd[i];
    }
    
    // پر کردن Bollinger Bands
    const filledBBands = Array(dataLength).fill({ upper: 0, middle: 0, lower: 0 });
    for (let i = 0; i < bbands.length; i++) {
      filledBBands[i + (dataLength - bbands.length)] = bbands[i];
    }
    
    // پر کردن ATR
    const filledATR = fillMissingValues(atr, dataLength);
    
    // پر کردن Stochastic
    const filledStochastic = Array(dataLength).fill({ k: 50, d: 50 });
    for (let i = 0; i < stochastic.length; i++) {
      filledStochastic[i + (dataLength - stochastic.length)] = stochastic[i];
    }

    // ساخت مجموعه داده ویژگی‌ها و برچسب‌ها
    for (let i = 0; i < dataLength; i++) {
      // مطمئن شویم که همه شاخص‌های تکنیکال برای این نقطه داده محاسبه شده‌اند
      if (i >= TRAINING_CONFIG.timeSteps) {
        // نرمال‌سازی قیمت‌ها با تقسیم بر قیمت فعلی
        const currentPrice = closes[i];
        
        // ویژگی‌های مربوط به این نقطه زمانی
        const featureRow = [
          opens[i] / currentPrice,  // قیمت باز شدن نرمال شده
          highs[i] / currentPrice,  // قیمت بالا نرمال شده
          lows[i] / currentPrice,   // قیمت پایین نرمال شده
          1.0,  // قیمت بسته شدن نرمال شده (همیشه 1.0 خواهد بود)
          volumes[i] / volumes[i-1],  // نسبت حجم
          filledSMA9[i] / currentPrice,  // SMA9 نرمال شده
          filledSMA20[i] / currentPrice,  // SMA20 نرمال شده
          filledEMA9[i] / currentPrice,  // EMA9 نرمال شده
          filledRSI[i] / 100,  // RSI نرمال شده (0-1)
          filledMACD[i].MACD / currentPrice * 100,  // MACD نرمال شده
          (filledBBands[i].upper - filledBBands[i].lower) / currentPrice,  // پهنای باند بولینگر نرمال شده
          filledATR[i] / currentPrice,  // ATR نرمال شده
          filledStochastic[i].k / 100  // Stochastic K نرمال شده (0-1)
        ];
        
        features.push(featureRow);
        
        // برچسب‌ها: تغییرات قیمت در افق‌های مختلف
        const priceChangeLabels = [];
        for (const horizon of TRAINING_CONFIG.predictionHorizons) {
          if (i + horizon < dataLength) {
            // تغییر قیمت نسبی در افق پیش‌بینی
            const futureChange = (closes[i + horizon] - closes[i]) / closes[i];
            priceChangeLabels.push(futureChange);
          } else {
            // اگر داده کافی برای افق پیش‌بینی نداریم، از مقدار صفر استفاده می‌کنیم
            priceChangeLabels.push(0);
          }
        }
        
        labels.push(priceChangeLabels);
      }
    }

    console.log(`${features.length} نمونه با ${features[0].length} ویژگی استخراج شد.`);
    return { features, labels };
  }

  // تبدیل داده‌ها به فرمت مناسب برای شبکه عصبی LSTM
  prepareSequenceData(features, labels) {
    const X = [];
    const y = [];
    
    console.log('در حال آماده‌سازی داده‌های توالی برای LSTM...');
    
    // ایجاد توالی‌های داده برای LSTM
    for (let i = TRAINING_CONFIG.timeSteps; i < features.length; i++) {
      const sequence = features.slice(i - TRAINING_CONFIG.timeSteps, i);
      X.push(sequence);
      y.push(labels[i - 1]); // استفاده از برچسب متناظر با آخرین نقطه داده در توالی
    }
    
    console.log(`${X.length} توالی آموزشی ایجاد شد.`);
    return { X, y };
  }

  // تقسیم داده‌ها به مجموعه‌های آموزش و اعتبارسنجی
  splitTrainValidation(X, y) {
    const splitIndex = Math.floor(X.length * (1 - TRAINING_CONFIG.validationSplit));
    
    const X_train = X.slice(0, splitIndex);
    const y_train = y.slice(0, splitIndex);
    const X_val = X.slice(splitIndex);
    const y_val = y.slice(splitIndex);
    
    console.log(`داده‌ها به ${X_train.length} نمونه آموزش و ${X_val.length} نمونه اعتبارسنجی تقسیم شدند.`);
    
    // تبدیل به تنسورهای TensorFlow.js
    const X_train_tensor = tf.tensor3d(X_train);
    const y_train_tensor = tf.tensor2d(y_train);
    const X_val_tensor = tf.tensor3d(X_val);
    const y_val_tensor = tf.tensor2d(y_val);
    
    return {
      trainData: [X_train_tensor, y_train_tensor],
      valData: [X_val_tensor, y_val_tensor]
    };
  }

  // ایجاد مدل LSTM
  createModel() {
    console.log('در حال ایجاد مدل LSTM...');
    
    const model = tf.sequential();
    
    // لایه LSTM اول با برگشت توالی
    model.add(tf.layers.lstm({
      units: 100,
      returnSequences: true,
      inputShape: [TRAINING_CONFIG.timeSteps, TRAINING_CONFIG.featureCount]
    }));
    
    // لایه Dropout برای جلوگیری از بیش‌برازش
    model.add(tf.layers.dropout({ rate: 0.2 }));
    
    // لایه LSTM دوم بدون برگشت توالی
    model.add(tf.layers.lstm({
      units: 50,
      returnSequences: false
    }));
    
    // لایه Dropout دیگر
    model.add(tf.layers.dropout({ rate: 0.2 }));
    
    // لایه کاملاً متصل
    model.add(tf.layers.dense({ units: 50, activation: 'relu' }));
    
    // لایه خروجی با تعداد افق‌های پیش‌بینی
    model.add(tf.layers.dense({
      units: TRAINING_CONFIG.predictionHorizons.length
    }));
    
    // کامپایل مدل
    model.compile({
      optimizer: tf.train.adam(TRAINING_CONFIG.learningRate),
      loss: 'meanSquaredError',
      metrics: ['mse', 'mae']
    });
    
    console.log(model.summary());
    return model;
  }

  // آموزش مدل
  async trainModel(model, trainData, valData, symbol) {
    console.log(`شروع آموزش مدل برای ${symbol}...`);
    
    // ذخیره بهترین مدل در طول آموزش
    const bestModelPath = path.join(MODELS_DIR, `${symbol.toLowerCase()}_best`);
    
    // مانیتور کردن پیشرفت آموزش و ذخیره بهترین مدل
    let bestValLoss = Infinity;
    
    // آموزش مدل
    await model.fit(trainData[0], trainData[1], {
      epochs: TRAINING_CONFIG.epochs,
      batchSize: TRAINING_CONFIG.batchSize,
      validationData: valData,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          console.log(`دوره ${epoch + 1}/${TRAINING_CONFIG.epochs}, loss: ${logs.loss.toFixed(6)}, val_loss: ${logs.val_loss.toFixed(6)}`);
          
          // ذخیره بهترین مدل بر اساس خطای اعتبارسنجی
          if (logs.val_loss < bestValLoss) {
            bestValLoss = logs.val_loss;
            await model.save(`file://${bestModelPath}`);
            console.log(`بهترین مدل تا این لحظه در دوره ${epoch + 1} با val_loss = ${bestValLoss.toFixed(6)} ذخیره شد.`);
          }
        }
      }
    });

    // بارگذاری بهترین مدل
    const bestModel = await tf.loadLayersModel(`file://${bestModelPath}/model.json`);
    console.log(`بهترین مدل از مسیر ${bestModelPath} بارگذاری شد.`);
    
    return bestModel;
  }

  // ارزیابی مدل
  evaluateModel(model, X_test, y_test) {
    console.log('در حال ارزیابی مدل...');
    
    // تبدیل به تنسور
    const X_test_tensor = tf.tensor3d(X_test);
    const y_test_tensor = tf.tensor2d(y_test);
    
    // ارزیابی مدل
    const evalResult = model.evaluate(X_test_tensor, y_test_tensor);
    console.log(`Test Loss: ${evalResult[0].dataSync()[0].toFixed(6)}`);
    console.log(`Test MSE: ${evalResult[1].dataSync()[0].toFixed(6)}`);
    console.log(`Test MAE: ${evalResult[2].dataSync()[0].toFixed(6)}`);
    
    // پیش‌بینی روی داده‌های آزمون
    const predictions = model.predict(X_test_tensor);
    const predValues = predictions.arraySync();
    const trueValues = y_test_tensor.arraySync();
    
    // محاسبه دقت جهت پیش‌بینی (رو به بالا یا رو به پایین)
    let correctDirections = 0;
    let totalPredictions = 0;
    
    for (let i = 0; i < predValues.length; i++) {
      for (let j = 0; j < predValues[i].length; j++) {
        // فقط پیش‌بینی‌های با مقدار غیر صفر را در نظر می‌گیریم
        if (trueValues[i][j] !== 0) {
          totalPredictions++;
          if ((predValues[i][j] > 0 && trueValues[i][j] > 0) || 
              (predValues[i][j] < 0 && trueValues[i][j] < 0)) {
            correctDirections++;
          }
        }
      }
    }
    
    const directionAccuracy = (correctDirections / totalPredictions) * 100;
    console.log(`دقت جهت پیش‌بینی: ${directionAccuracy.toFixed(2)}%`);
    
    return {
      mse: evalResult[1].dataSync()[0],
      mae: evalResult[2].dataSync()[0],
      directionAccuracy
    };
  }

  // تابع اصلی آموزش برای یک ارز دیجیتال
  async trainForCrypto(crypto) {
    try {
      console.log(`=== شروع آموزش برای ${crypto.name} (${crypto.symbol}) ===\n`);
      
      // دریافت داده‌های تاریخی
      const historicalData = await this.fetchHistoricalData(crypto.symbol, '1h', 5000);
      
      // استخراج ویژگی‌ها
      const { features, labels } = this.extractFeatures(historicalData);
      
      // آماده‌سازی داده‌ها برای LSTM
      const { X, y } = this.prepareSequenceData(features, labels);
      
      // تقسیم به داده‌های آموزش و اعتبارسنجی
      const { trainData, valData } = this.splitTrainValidation(X, y);
      
      // ایجاد مدل
      const model = this.createModel();
      
      // آموزش مدل
      const trainedModel = await this.trainModel(model, trainData, valData, crypto.symbol);
      
      // ارزیابی مدل
      const testResults = this.evaluateModel(trainedModel, X.slice(-100), y.slice(-100));
      
      // ذخیره مدل نهایی
      const modelSavePath = path.join(MODELS_DIR, crypto.symbol.toLowerCase());
      await trainedModel.save(`file://${modelSavePath}`);
      console.log(`مدل نهایی در مسیر ${modelSavePath} ذخیره شد.\n`);
      
      // ذخیره اطلاعات مدل
      const modelInfo = {
        symbol: crypto.symbol,
        name: crypto.name,
        lastTrainingDate: new Date().toISOString(),
        performance: testResults,
        config: TRAINING_CONFIG,
        features: TRAINING_CONFIG.featureCount,
        timeSteps: TRAINING_CONFIG.timeSteps,
        predictionHorizons: TRAINING_CONFIG.predictionHorizons
      };
      
      fs.writeFileSync(
        path.join(modelSavePath, 'info.json'),
        JSON.stringify(modelInfo, null, 2)
      );
      
      console.log(`=== آموزش برای ${crypto.name} (${crypto.symbol}) با موفقیت به پایان رسید ===\n`);
      
      return {
        symbol: crypto.symbol,
        success: true,
        performance: testResults
      };
    } catch (error) {
      console.error(`خطا در آموزش مدل برای ${crypto.symbol}:`, error);
      return {
        symbol: crypto.symbol,
        success: false,
        error: error.message
      };
    }
  }

  // آموزش مدل برای همه ارزهای دیجیتال
  async trainAllModels() {
    console.log('=== شروع آموزش برای تمام ارزهای دیجیتال ===\n');
    
    const results = [];
    
    for (const crypto of CRYPTOS) {
      const result = await this.trainForCrypto(crypto);
      results.push(result);
    }
    
    console.log('=== نتایج آموزش برای تمام ارزهای دیجیتال ===');
    console.table(results.map(r => ({
      symbol: r.symbol,
      success: r.success,
      mse: r.success ? r.performance.mse.toFixed(6) : '-',
      directionAccuracy: r.success ? `${r.performance.directionAccuracy.toFixed(2)}%` : '-',
      error: r.success ? '-' : r.error
    })));
    
    console.log('\n=== آموزش برای تمام ارزهای دیجیتال با موفقیت به پایان رسید ===');
    
    return results;
  }
}

// اجرای برنامه آموزش
async function main() {
  try {
    console.log('🤖 برنامه آموزش مدل هوش مصنوعی تحلیل‌گر ارز دیجیتال\n');
    
    const trainer = new ModelTrainer();
    await trainer.trainAllModels();
    
  } catch (error) {
    console.error('خطای اصلی در برنامه آموزش:', error);
  }
}

// اجرای برنامه
main();
