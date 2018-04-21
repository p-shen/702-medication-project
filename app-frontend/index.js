/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as ui from './ui';

const URLS = {
  url: 'http://0.0.0.0:5000/predict/',
};

class SentimentPredictor {
  /**
   * Initializes the Sentiment demo.
   */
  async init(urls) {
    this.urls = urls.url;
    return this;
  }

  predict(text) {

    // Convert to lower case and remove all punctuations.
    const inputText =
        text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');

    ui.status('Running inference');

    const beginMs = performance.now();


    var formData = new FormData();
    formData.append("data", inputText);

    var xmlhttp = new XMLHttpRequest();
    xmlhttp.open("POST", 'http://localhost:5000/predict/', false);
    xmlhttp.send(formData);
    var response = xmlhttp.responseText
    const endMs = performance.now();

    return {drug: response, elapsed: (endMs - beginMs)};
  }
};


/**
 * Loads the pretrained model and metadata, and registers the predict
 * function with the UI.
 */
async function setupSentiment() {
  const predictor = await new SentimentPredictor().init(URLS);
  ui.prepUI(x => predictor.predict(x));
  // ui.status('Input clinical text.');
}

setupSentiment();
