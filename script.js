/**
 * Fetch car data and extract data we're interested in
 **/

const getData = async () => {
  const carsDataRes = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
  if (!carsDataRes.ok) {
    throw new Error("Failed to fetch");
  }

  const carsData = await carsDataRes.json();
  return carsData
    .map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
      cylinders: car.Cylinders,
    }))
    .filter(x => x.mpg != null && x.horsepower != null);
};

const tensorize = data => {
  tf.util.shuffle(data);

  const inputs = data.map(d => [d.horsepower, d.cylinders]);
  const outputs = data.map(d => d.mpg);

  const inputTensor = tf.tensor2d(
    inputs,
    [inputs.length, 2],
  );

  const labelTensor = tf.tensor2d(
    outputs,
    [outputs.length, 1],
  );

  const inputMax = tf.transpose(inputTensor).max(1);
  const inputMin = tf.transpose(inputTensor).min(1);
  const labelMax = labelTensor.max();
  const labelMin = labelTensor.min();

  const normalizedInputs = inputTensor
    .sub(inputMin)
    .div(inputMax.sub(inputMin));

  const normalizedLabels = labelTensor
    .sub(labelMin)
    .div(labelMax.sub(labelMin));

  return {
    inputs: normalizedInputs,
    labels: normalizedLabels,

    inputMax,
    inputMin,
    labelMax,
    labelMin,
  };
};

const getStronger = (model, inputs, labels) => {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = 10;

  return model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] },
    ),
  });
};

const createModel = () => {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [2],
    units: 100,
  }));
  model.add(tf.layers.dense({
    units: 100,
    activation: 'sigmoid',
  }));
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid',
  }));
  return model;
};

const testModel = (model, inputData, { inputMax, inputMin, labelMin, labelMax }) => {
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    const newXs = tf.stack([
      xs,
      xs
    ]);

    const preds = model.predict(newXs.transpose());
  
    preds.print();

    inputMax.max().print();
    inputMin.min().print();
    labelMax.print();
    labelMin.print();

    const unNormXs = xs 
      .mul(inputMax.max().sub([40]))
      .add([40]);
    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => ({ x: val, y: preds[i] }));
  const originalPoints = inputData.map(d => ({ x: d.horsepower, y: d.mpg }));

  tfvis.render.scatterplot(
    { name: 'Model Predicitions vs Original Data' },
    { 
      values: [originalPoints, predictedPoints],
      series: ['original', 'predicted'],
    },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300,
    },
  );
};

const run = async () => {
  const data = await getData();
  
  const model = createModel();
  tfvis.show.modelSummary(
    { name: 'Model Summary' },
    model,
  );

  const tensorData = tensorize(data);
  const { inputs, labels, inputMax, inputMin, labelMax, labelMin } = tensorData;
  await getStronger(model, inputs, labels);
  
  testModel(model, data, tensorData);
};

document.addEventListener('DOMContentLoaded', run);

