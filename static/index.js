const img = document.querySelector("img");
let lastModel;
let lastClassName;
let warmedUp = false;

setupButton(
  "tfjs-webgl",
  async () =>
    await tfTask.ObjectDetection.CocoSsd.TFJS.load({
      backend: "webgl"
    }),
  true
);

setupButton(
  "tfjs-wasm",
  async () =>
    await tfTask.ObjectDetection.CocoSsd.TFJS.load({
      backend: "wasm"
    }),
);

setupButton(
  "tflite",
  async () => await tfTask.ObjectDetection.CocoSsd.TFLite.load()
);

setupButton(
  "tflite-custom",
  async () =>
    await tfTask.ObjectDetection.CustomModel.TFLite.load({
      model:
        "https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2?lite-format=tflite"
    })
);

async function setupButton(className, modelCreateFn, needWarmup) {
  document
    .querySelector(`.model.${className} .btn`)
    .classList.remove("disabled");
  const resultEle = document.querySelector(`.model.${className} .result`);
  document
    .querySelector(`.model.${className} .btn`)
    .addEventListener("click", async () => {
      let model;
      // Create the model when user clicks on a button.
      if (lastClassName !== className) {
        // Clean up the previous model if existed.
        if (lastModel) {
          lastModel.cleanUp();
        }
        // Create the new model and save it.
        resultEle.textContent = "Loading...";
        model = await modelCreateFn();
        lastModel = model;
        lastClassName = className;
      }
      // Reuse the model if user clicks on the same button.
      else {
        model = lastModel;
      }

      // Warm up if needed.
      if (needWarmup && !warmedUp) {
        await model.predict(img);
        warmedUp = true;
      }

      // Run inference and update result.
      const start = Date.now();
      const result = await model.predict(img);
      const latency = Date.now() - start;
      renderDetectionResult(result);
      resultEle.textContent = `Latency: ${latency}ms`;
    });
}

/** Render detection results. */
function renderDetectionResult(result) {
  const boxesContainer = document.querySelector(".boxes-container");
  boxesContainer.innerHTML = "";
  const objects = result.objects;
  for (let i = 0; i < Math.min(5, objects.length); i++) {
    const curObject = objects[i];
    const boundingBox = curObject.boundingBox;
    const name = curObject.className;
    const score = curObject.score;

    const boxContainer = createDetectionResultBox(
      boundingBox.originX,
      boundingBox.originY,
      boundingBox.width,
      boundingBox.height,
      name,
      score
    );
    boxesContainer.appendChild(boxContainer);
  }
}

/** Create a single detection result box. */
function createDetectionResultBox(left, top, width, height, name, score) {
  const container = document.createElement("div");
  container.classList.add("box-container");

  const box = document.createElement("div");
  box.classList.add("box");
  container.appendChild(box);

  const label = document.createElement("div");
  label.classList.add("label");
  label.textContent = `${name} (${score.toFixed(2)})`;
  container.appendChild(label);

  container.style.left = `${left - 1}px`;
  container.style.top = `${top - 1}px`;
  box.style.width = `${width + 1}px`;
  box.style.height = `${height + 1}px`;

  return container;
}
