import * as tf from '@tensorflow/tfjs';


import {assertPositiveInteger, getRandomInteger} from './utils';


// TODO(cais): Tune these parameters.
export const SCORE_REWARD = 2;
export const DEATH_REWARD = -10;

export const ACTION_NONE = 0;
export const ACTION_LEFT = 1;
export const ACTION_RIGHT = 2;

export const ALL_ACTIONS = [ACTION_NONE, ACTION_LEFT, ACTION_RIGHT];
export const NUM_ACTIONS = ALL_ACTIONS.length;

/**
 * Generate a random action among all possible actions.
 *
 * @return {0 | 1 | 2} Action represented as a number.
 */
export function getRandomAction() {
    return getRandomInteger(0, NUM_ACTIONS);
}

export class FlyGame {

    constructor(args) {
        this.canvas = args.canvas || null;
        this.width = args.width || 100;
        this.height = args.height || 100;
        this.initSpeed = args.initSpeed || 1;
        this.scale = args.scale || 1;
        this.enemySpawns = args.enemySpawns || 3;

        if (this.canvas) {
            this.canvas.width = this.width;
            this.canvas.height = this.height;
            this.canvas.style.width = `${this.width * this.scale}px`;
            this.canvas.style.height = `${this.height * this.scale}px`;
    
            this.canvas.imageSmoothingEnabled = false;
    
            this.ctx = this.canvas.getContext('2d');
        }

        this.reset();
    }

    reset() {
        this.enemies = [];
        this.coins = null;
        this.speed = this.initSpeed;
        this.score = 0;
        this.position = {x: this.width / 2, y: this.height - 2};

        //add initial enemy state...
        for (let i = this.height-4; i > -1; i--) {
            this.PATH_X = (this.width / 2) + Math.sin((i / this.height) - 4) * (this.width / 3);
            this.addEnemyRow(this.PATH_X, i);
        }
    }

    addCoin() {
        if (this.coins == null) {
            this.coins = [];
        }
    }

    addEnemyRow(x, y) {
        for (let i = 0; i < this.width; i++) {
            if (i > x + 2 || i < x - 2) {
                this.enemies.push([i, y]); 
            }
        }
    }

    addEnemy() {
        if (this.enemies == null) {
            this.enemies = [];
        }

        this.enemies.push([Math.floor(Math.random() * this.width), 0]);
    }

    /**
   * Perform a step of the game.
   *
   * @param {0 | 1 | 2} action The action to take in the current step.
   *   The meaning of the possible values:
   *     0 - none
   *     1 - left
   *     2 - right
   * @return {object} Object with the following keys:
   *   - `reward` {number} the reward value.
   *     - 0 if no fruit is eaten in this step
   *     - 1 if a fruit is eaten in this step
   *   - `state` New state of the game after the step.
   *   - `fruitEaten` {boolean} Whether a fruit is easten in this step.
   *   - `done` {boolean} whether the game has ended after this step.
   *     A game ends when the head of the snake goes off the board or goes
   *     over its own body.
   */
  //{state: nextState, reward, done, score}
    step(action) {
        let training = action !== undefined ? true : false;

        if (action == ACTION_LEFT) {
            this.position.x -= 1;
        } else if (action == ACTION_RIGHT) {
            this.position.x += 1;
        }

        this.score++;


        let dir = 0;
        if (this.score % 5 == 0) {
            dir = Math.random() > 0.5 ? 1 : -1;
            if (this.PATH_X + dir > this.width) {
                dir = -1;
            } else if (this.PATH_X + dir < 0) {
                dir = 1;
            }
        }

        this.PATH_X += dir;
        this.addEnemyRow(this.PATH_X, -1);

        if (!this.enemies) {
            return;
        }

        if (this.position.x > this.width - 1 || this.position.x < 0) {
            if (training) {
                return {reward: DEATH_REWARD, done: true, score: this.score};
            } else {
                this.gameover();
                return;
            }
        } 

        for (let i = this.enemies.length - 1; i > 0 ; i--) {
            let enemy = this.enemies[i];

            if (enemy[1] == this.position.y && enemy[0] == this.position.x) { //game over..
                if (training) {
                    return {reward: DEATH_REWARD, done: true, score: this.score};
                } else {
                    this.gameover();
                    return;
                }
            }

            enemy[1] += this.speed;

            if (enemy[1] > this.height) {
                this.enemies.splice(i, 1);
            }
        }

        if (training) {
            const state = this.getState();

            return {reward: SCORE_REWARD * this.score, state, done: false, score: this.score};
        }

    }

    play() {
        this.keyHandler = (e) => this.handle_KEY_PRESS(e);

        this.int1 = setInterval(() => this.step(), 100);
        this.int2 = setInterval(() => this.render(), 100);
        window.addEventListener('keydown', this.keyHandler)
    }

    stop() {
        clearInterval(this.int1);
        clearInterval(this.int2);
        window.removeEventListener('keydown', this.keyHandler);
    }

    handle_KEY_PRESS(e) {
        if (e.key == 'ArrowLeft') {
            this.position.x --;
        } 
        if (e.key == 'ArrowRight') {
            this.position.x ++;
        }
    }

    render() {
        if (!this.canvas) {
            return;
        }
        this.ctx.clearRect(0, 0, this.width, this.height);

        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(this.position.x, this.position.y, 1, 1);

        for (let i = this.enemies.length - 1; i > 0 ; i--) {
            let enemy = this.enemies[i];
            this.ctx.fillStyle = 'red';
            this.ctx.fillRect(enemy[0], enemy[1], 1, 1);
        }

        scoreElement.innerText = `Score=${this.score}`;
    }

    gameover() {
        console.log('gameover');
        this.reset();
    }

    /**
   * Get plain JavaScript representation of the game state.
   *
   * @return An object with two keys:
   *   - s: {Array<[number, number]>} representing the squares occupied by
   *        the snake. The array is ordered in such a way that the first
   *        element corresponds to the head of the snake and the last
   *        element corresponds to the tail.
   *   - f: {Array<[number, number]>} representing the squares occupied by
   *        the fruit(s).
   */
    getState() {
        return {
            "p": [this.position.x, this.position.y],
            "e": this.enemies.slice()
        }
    }
}


/**
 * Get the current state of the game as an image tensor.
 *
 * @param {object | object[]} state The state object as returned by
 *   `SnakeGame.getState()`, consisting of two keys: `s` for the snake and
 *   `f` for the fruit(s). Can also be an array of such state objects.
 * @param {number} h Height.
 * @param {number} w With.
 * @return {tf.Tensor} A tensor of shape [numExamples, height, width, 2] and
 *   dtype 'float32'
 *   - The first channel uses 0-1-2 values to mark the snake.
 *     - 0 means an empty square.
 *     - 1 means the body of the snake.
 *     - 2 means the haed of the snake.
 *   - The second channel uses 0-1 values to mark the fruits.
 *   - `numExamples` is 1 if `state` argument is a single object or an
 *     array of a single object. Otherwise, it will be equal to the length
 *     of the state-object array.
 */

export function getStateTensor(state, h, w) {
    if (!Array.isArray(state)) {
      state = [state];
    }
    const numExamples = state.length;
    // TODO(cais): Maintain only a single buffer for efficiency.
    const buffer = tf.buffer([numExamples, h, w, 2]);
  
    for (let n = 0; n < numExamples; ++n) {
      if (state[n] == null) {
        continue;
      }

      //mark the player...
      state[n].p.forEach(yx => {
        buffer.set(1, n, yx[0], yx[1], 0);
      });

      //mark the enemies...
      state[n].e.forEach(yx => {
        buffer.set(1, n, yx[0], yx[1], 1);
      });
    }
    return buffer.toTensor();
  }
  