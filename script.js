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
  const data = (await getData())
    .map(d => ({
      x: d.horsepower,
      y: d.mpg,
    }));

  tfvis.render.scatterplot(
    { name: 'Horsepower vs. MPG' },
    { values: data },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );

  const model = createModel();
  tfvis.show.modelSummary(
    { name: 'Model Summary' },
    model,
  );
};

document.addEventListener('DOMContentLoaded', run);

