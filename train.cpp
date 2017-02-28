#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <dirent.h>
using namespace std;

const int digit_vec = 784;
const int hiddlen_num = 100;
const int result_num = 10;
const double alpha = 0.35;

int target[result_num];
int input[digit_vec];
double output1[hiddlen_num];
double output2[result_num];
double weight1[digit_vec][hiddlen_num];
double weight2[hiddlen_num][result_num];
double b1[hiddlen_num],b2[result_num];
double deta1[hiddlen_num],deta2[result_num];

void init(){
    srand((int)time(0));
    for(int i = 0;i < digit_vec; i ++){
        for(int j = 0;j < hiddlen_num; j ++){
            weight1[i][j] = (rand() % 1000 - 500) / 1000.0;
        }
    }

    for(int i = 0;i < hiddlen_num; i ++){
        for(int j = 0;j < result_num; j ++){
            weight2[i][j] = (rand() % 1000 - 500) / 1000.0;
        }
    }
    for(int i = 0; i < hiddlen_num; i ++)
      b1[i] =  (rand() % 1000 - 500) / 1000.0;
    for(int i = 0;i < result_num; i ++)
      b2[i] =  (rand() % 1000 - 500) / 1000.0;

    return ;
}

double sigmod(double val){
    return 1.0 /(1.0 + exp(-val));
}

//正向传播，输入层到隐藏层
void op1(){

    for(int k = 0; k < 100; k ++){
        double sum = 0;
        for(int j = 0; j < 784; j++){
            sum += input[j] * weight1[j][k];
        }
        output1[k] =sigmod(b1[k] + sum);
    }

    return;
}

//正向传播，隐藏层到输出层
void op2(){

    for(int k = 0; k < 10; k ++){
        double sum = 0;
        for(int j = 0; j < 100; j++){
            sum += output1[j] * weight2[j][k];
        }
        output2[k] =sigmod(b2[k] + sum);
    }
    return ;
}

//计算输出层向量的误差的梯度

void dt_op2(){

    for(int i = 0; i < 10; i ++){
        deta2[i] = output2[i] * (1 -output2[i]) * (output2[i] - target[i]);
    }
    return;
}

//计算隐藏层向量的误差的梯度

void dt_op1(){
    for(int k = 0; k < 100; k ++){
        double sum = 0;
        for(int j = 0; j < 10;j ++){
            sum +=  weight2[j][k] * deta2[j];
        }
        deta1[k] = output1[k] * (1 - output1[k]) * sum;
    }
    return ;
}

//更新输入层到隐藏层权值参数
void feedback_hiddlen(){
    
    for(int k = 0; k < 100; k ++){
        b1[k] = b1[k] - alpha * deta1[k];
        for(int j = 0; j < 784; j ++){
            weight1[j][k] = weight1[j][k] - alpha * input[j] * deta1[k];
        }
    }
    return ;
}

//更新隐藏层到输出层权值参数
void feedback_output(){
    for(int k = 0; k < 10; k ++){
        b2[k] = b2[k] - alpha * deta2[k];
        for(int j = 0; j < 100; j ++){
            weight2[j][k] = weight2[j][k] - alpha * output1[j] * deta2[k];
        }
    }
    return ;
}

//保存模型参数
void save_model(){
    if(opendir("./data") == NULL){
        mkdir("./data", 0775);
    }
    FILE *f = fopen("./data/pkl", "w");
    fwrite(b1, sizeof(double), 100, f);
    fwrite(b2, sizeof(double), 10, f);
    for(int i = 0;i < 784; i ++)
      fwrite(weight1[i], sizeof(double), 100, f);
    for(int i = 0;i < 100;i ++)
      fwrite(weight2[i], sizeof(double), 10, f);
    fclose(f);    
    return ;
}

void train(){
    int cnt = 0;
    FILE * f_train_x;
    FILE * f_train_y;
    f_train_x = fopen("./tc/train-images.idx3-ubyte", "rb");
    f_train_y = fopen("./tc/train-labels.idx1-ubyte", "rb");
    unsigned char train_x[digit_vec],train_y[result_num];
    int useless[1000];
    fread(useless, 1, 16, f_train_x);
    fread(useless, 1, 8, f_train_y);
    
    while(!feof(f_train_x) && !feof(f_train_y)){
        memset(train_x, 0, 784);
        memset(train_y, 0, 10);
        fread(train_x,1,784,f_train_x);
        fread(train_y,1,1,f_train_y);
        for(int i = 0;i < 784; i ++){
            if((unsigned int)train_x[i] < 128){
                input[i] = 0;
            }
            else{
                input[i] = 1;
            }
        }
        for(int k = 0; k < 10; k ++){
            target[k] = 0;
        }
        int label_value = (unsigned int)train_y[0];
        target[label_value] = 1;
        op1();
        op2();
        dt_op2();
        dt_op1();
        feedback_hiddlen();
        feedback_output();
        cnt ++;
        if(cnt % 1000 == 0){
            cout << "train digit :" << cnt <<endl;
        }
    }

    cout << "train finish !!!!" << endl << "start save model!" << endl;
    save_model();
    return;
}

int main(){
    init();
    cout << "init success" << endl;
    train();
    cout << "train success, you can run ./predict " << endl;
    return 0;
}
