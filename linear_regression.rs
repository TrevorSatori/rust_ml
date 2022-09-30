use ndarray::prelude::*;
use std::{io, result};

fn main(){

    // input variable (home size in 1k square feet)
    let x_train: Array2<f64> = array![[2104.0, 5.0, 1.0, 45.0], [1416.0, 3.0, 2.0, 40.0], [852.0, 2.0, 1.0, 35.0]];
    let y_train: Array1<f64> = array![460.0, 232.0, 178.0];

    // print dimensions
    // println!("X_train dim: {:?}", &x_train.ndim());
    // println!("Y_train dim: {:?}", &y_train.ndim());

    // println!("input (x) shape: {:?}", x_train.shape());

    // weights and bias
    let mut b_init: f64 = 0.0;
    let mut w_init: Array1<f64> = array![ 0.0, 0.0, 0.0, 0.0];
    
    //predict_single_loop(&x_train, &w_init, &b_init);
    //let a = mean_squared_error(&x_train, &y_train, &w_init, &b_init);
    //println!("{:?}", a);

    // let (b, w) = compute_gradient(&x_train, &y_train, &w_init, &b_init);
    
    // println!("Weights: {:?}", weights);

    let alpha = 0.0000005;
    let iters = 1000;
    let (bias,weights) = gradient_descent(&x_train, &y_train, w_init, b_init, &alpha, &iters);
    println!("bias: {:?}, weights: {:?}", bias, weights);
}


fn predict_single_loop(x: &Array2<f64>, w: &Array1<f64>, b: &f64){
    // inputs * weights of each associated feature
    let dot = x.row(0).dot(w) + b;
    println!("{:?}", dot);
}

fn mean_squared_error(x: &Array2<f64>, y: &Array1<f64>, w: &Array1<f64>, b: &f64) -> f64{

    // number of samples (rows)
    let m = x.shape()[0];
    
    let mut cost = 0.0;

    for i in 0..m {

        // prediction (x_sub_0 * w_sub_0 .. x_sub_n * w_sub_n)
        // multiplies all inputs with respective weights for m samples 
        let pred = x.row(i).dot(w) + b;

        //predictin - actual squared
        *&mut cost += (pred - y[i]).powf(2.0);
    }
    return cost / (2.0 * m as f64);
}

// calculate partial derivative
fn compute_gradient(x: &Array2<f64>, y: &Array1<f64>, w: &Array1<f64>, b: &f64) -> (f64, Array1<f64>){
    
    // number of samples
    let m = x.shape()[0];

    // number of input features
    let n = x.shape()[1];

    let mut dj_dw = Array1::<f64>::zeros(n);
    let mut dj_db: f64 = 0.0;

    //for samples (rows)
    for i in 0..m{

        // get prediction vector for all (x) features then subtract by actual
        let error: f64 = (x.row(i).dot(w) + b) - y[i];

        // calculate partial derivative over features
        for j in 0..n{
            *&mut dj_dw[j] += error * x[[i, j]] as f64;
        }
        // update bias
        *&mut dj_db += error;

    }
    let dj_dw = dj_dw / m as f64;
    let dj_db = dj_db / m as f64;

    //println!("derivative weights: {:?} Derivative Bias: {:?}", dj_dw, dj_db);
    return (dj_db, dj_dw);
}   

fn gradient_descent(x: &Array2<f64>, y: &Array1<f64>, w_in: Array1<f64>, b_in: f64, alpha: &f64, iters: &u32) -> (f64, Array1<f64>){
    
    let mut w = w_in.clone();
    let mut b = b_in.clone();

    // iterate through gradient descent process
    for i in 0..*iters{

        let (dj_db, dj_dw) = compute_gradient(&x, &y, &w, &b);
        // println!("{:?}", &w);

        w = w - (*alpha * dj_dw);
        b = b - (*alpha * dj_db);


        if i % 1000 == 0 {
            println!("iteration: {:?}, cost: {:?}", i, mean_squared_error(&x, &y, &w, &b))
        }
    }

    return (b,w);
}

fn reader() {
    let mut rdr = csv::Reader::from_reader(io::stdin());
    for result in rdr.records(){
        let record = result.expect("a csv record");
        println!("{:?}", record);
    }
}
