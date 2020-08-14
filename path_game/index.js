
import * as tf from '@tensorflow/tfjs';

import {ALL_ACTIONS, getStateTensor, FlyGame} from './game';

const LOCAL_MODEL_URL = './models/batch3/100_10x10/model.json';


let game;
let qNet;

let cumulativeReward = 0;
let totalScore = 0;
let autoPlayIntervalJob;

/** Reset the game state. */
async function reset() {
  if (game == null) {
    return;
  }
  game.reset();
  await calcQValuesAndBestAction();
  game.render();
  /*
  renderSnakeGame(gameCanvas, game,
      showQValuesCheckbox.checked ? currentQValues : null);
  gameStatusSpan.textContent = 'Game started.';
  stepButton.disabled = false;
  autoPlayStopButton.disabled = false;
  */

}

/**
 * Play a game for one step.
 *
 * - Use the current best action to forward one step in the game.
 * - Accumulate to the cumulative reward.
 * - Determine if the game is over and update the UI accordingly.
 * - If the game has not ended, calculate the current Q-values and best action.
 * - Render the game in the canvas.
 */
async function step() {
    const {reward, done, score} = game.step(bestAction);
    invalidateQValuesAndBestAction();
    cumulativeReward += reward;
    totalScore = score;
    scoreElement.innerText =
        `Reward=${cumulativeReward.toFixed(1)}; Score=${totalScore}`;
    if (done) {
      console.log("DONE>>>");
      //gameStatusSpan.textContent += '. Game Over!';
      cumulativeReward = 0;
      //cumulativeFruits = 0;
      totalScore = 0;
      if (autoPlayIntervalJob) {
        clearInterval(autoPlayIntervalJob);
        //autoPlayStopButton.click();
      }
      //autoPlayStopButton.disabled = true;
      //stepButton.disabled = true;
    }
    await calcQValuesAndBestAction();
    //renderSnakeGame(gameCanvas, game,
    //    showQValuesCheckbox.checked ? currentQValues : null);
    game.render();
  }


let currentQValues;
let bestAction;

/** Calculate the current Q-values and the best action. */
async function calcQValuesAndBestAction() {
  if (currentQValues != null) {
    return;
  }
  tf.tidy(() => {
    const stateTensor = getStateTensor(game.getState(), game.height, game.width);
    const predictOut = qNet.predict(stateTensor);
    currentQValues = predictOut.dataSync();
    bestAction = ALL_ACTIONS[predictOut.argMax(-1).dataSync()[0]];
  });
}

function invalidateQValuesAndBestAction() {
  currentQValues = null;
  bestAction = null;
}

function initGame() {
    game = new FlyGame({
        canvas: document.getElementById('mygame'),
        width: 10,
        height: 10,
        scale: 10,
        enemySpawns: 2
    });
}

async function initQNET() {
    // Warm up qNet.
    for (let i = 0; i < 3; ++i) {
        qNet.predict(getStateTensor(game.getState(), game.height, game.width));
    }

    await reset();
}

(async function() {

    initGame();

    startBtn.addEventListener('click', () => {
      game.stop();
      game.play();
    });

    try {

      qNet = await tf.loadLayersModel(LOCAL_MODEL_URL);

      initQNET();

      autoplayBtn.addEventListener('click', () => {
        game.reset();
        autoPlayIntervalJob = setInterval(() => {
            step(game, qNet);
          }, 100);
      });

    } catch (err) {
      console.log('Loading local model failed.');
    }
  
  })();