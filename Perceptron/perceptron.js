class Perceptron{
    weights = [];
    lr = 0.05;
    consecutiveHints = 0;

    constructor(numberWeigths){
        this.weights = new Array(numberWeigths)
        this.lastWeights = new Array(numberWeigths);
        for(let i = 0;  i < this.weights.length; i++){
            this.weights[i] = random(-1,1);
            this.lastWeights[i] = this.weights[i];
        }
    }

    train(input, target) {
        const guess = this.guess(input);
        const error = target - guess;
        for(let i = 0;  i < this.weights.length; i++){
            this.weights[i] += error * input[i] * this.lr;
        }

        if (error == 0) {
            this.consecutiveHints++;
        }
        if (error != 0 ) {
            this.consecutiveHints = 0;
        }
    }

    guess(inputs){
        let sum = 0;

        for(let i = 0; i < this.weights.length; i++){
            sum += inputs[i] * this.weights[i];
        }
        const output = this.sign(sum);
        
        return output;
    }

    guessY(x) {
        const w0 = this.weights[0];
        const w1 = this.weights[1];
        const w2 = this.weights[2];

        return -(w2 / w1) - (w0 / w1) * x;
    }

    sign(num){
        return num >= 0 ? 1 : -1;
    }


    // Função que verifica se os pesos mudaram
    // weightsChanged() {
    //     for (let i = 0; i < this.weights.length; i++) {
    //         if (this.weights[i] !== this.lastWeights[i]) {
    //             return true;
    //         }
    //     }
    //     return false;
    // }
}