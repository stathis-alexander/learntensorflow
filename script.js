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
    }))
    .filter(x => x.mpg != null && x.horsepower != null);
};

const tensorize = data => {
  tf.util.shuffle(data);

  const inputs = data.map(d => d.horsepower);
  const outputs = data.map(d => d.mpg);

  const inputTensor = tf.tensor2d(
    inputs,
    [inputs.length, 1],
  );

  const labelTensor = tf.tensor2d(
    outputs,
    [outputs.length, 1],
  );

  const inputMax = inputTensor.max();
  const inputMin = inputTensor.min();
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
  const epochs = 50;

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
    inputShape: [1],
    units: 1,
  }));
  model.add(tf.layers.dense({
    units: 1,
  }));
  return model;
};

const run = async () => {
  const data = await getData();
  
  const model = createModel();
  tfvis.show.modelSummary(
    { name: 'Model Summary' },
    model,
  );

  const { inputs, labels } = tensorize(data);
  await getStronger(model, inputs, labels);
  
  console.log('Harder, better, faster, stronger.')
};

document.addEventListener('DOMContentLoaded', run);

