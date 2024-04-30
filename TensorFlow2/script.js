async function getData() {
    const carsDataResponse = await
    fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map( car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower
    }))
    .filter( car => (car.mpg != null && car.horsepower != null));

    return cleaned;
}

async function run() {
    const data = await getData();
    const values = data.map( d => ({
        x: d.horsepower,
        y: d.mpg
    }));

    tfvis.render.scatterplot(
        {name: "Horsepower vs MilesPerGallon"},
        {values},
        {
            xLabel: "Horsepower",
            yLabel: "MPG",
            height: 300
        }
    );

    // Converte os dados
    const tensorData = convertToTensor(data);
    const { inputs, labels } = tensorData;

    // Treina o modelo
    await trainModel(model, inputs, labels);
    console.log("Treino completo!");

    await testModel(model, data, tensorData);
    console.log("Gráfico de resultado.");
}


document.addEventListener("DOMContentLoaded", run);

function createModel() {
    // Cria um modelo sequencial
    const model = tf.sequential();

    // Adiciona uma única camada de entrada
    model.add(tf.layers.dense( {inputShape: [1], units: 50, useBias: true} ));

    model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}))

    // Adiciona uma camada de saída
    model.add(tf.layers.dense( {units: 1, useBias: true} ));

    return model;
}



function convertToTensor(data) {
    
    return tf.tidy( () => {
        // Passo 1: embaralhe os dados
        tf.util.shuffle(data);
        
        // Passo 2: converta dados em tensor
        const inputs = data.map((d) => d.horsepower);
        const labels = data.map((d) => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelsTensor = tf.tensor2d(labels, [labels.length, 1]);

        // Passo 3: Normalize os dados para o intervalo 0 - 1 usando escala min-max
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelsTensor.max();
        const labelMin = labelsTensor.min();

        // Essa operação é a escala min-max (retorna um número entre 0 e 1)
        const normalizedInputs = inputTensor
            .sub(inputMin)
            .div(inputMax.sub(inputMin));

        const normalizedLabels = labelsTensor
            .sub(labelMin)
            .div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Retorne os limites mínimo/máximo para que posamos usá-los mais tarde.
            inputMax: inputMax,
            inputMin: inputMin,
            labelMax: labelMax,
            labelMin: labelMin
        };

    });
}

const model = createModel();
tfvis.show.modelSummary({name: "Modelo"}, model);


async function trainModel(model, inputs, labels) {
    // Prepara o modelo para o treinamento.
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ["mse"]
    });

    const batchSize = 32;
    const epochs = 50;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: "Performance do treinamento" },
            [ "loss", "mse" ],
            { height: 200, callbacks: ["onEpochEnd"] }
        ),
    });
}

function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax } = normalizationData;

    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0,1,100);
        const preds = model.predict(xs.reshape([100, 1]));

        const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

        const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
        return {x: val, y: preds[i]};
    });

    const originalPoints = inputData.map(d => ({
        x: d.horsepower,
        y: d.mpg
    }));

    tfvis.render.scatterplot(
        {name: "Previsões vs Dados originais"},
        {
            values: [originalPoints, predictedPoints],
            series: ["original", "predicted"]
        },
        {
            xLabel: "Horsepower",
            yLabel: "MPG",
            height: 300
        }
    )
}