#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
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
//double deta1[hiddlen_num],deta2[result_num];


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

void init_model(){
    FILE *f = fopen("./data/pkl", "r");
    cout << "开始加载权值数据";
    fread(b1, sizeof(double), 100, f);
    fread(b2, sizeof(double), 10, f);
    for(int i = 0;i < 784;i ++){
        fread(weight1[i], sizeof(double), 100, f);
    }
    for(int i = 0;i < 100;i ++){
        fread(weight2[i], sizeof(double), 10, f);
    }
    fclose(f);
    cout << endl;
    cout << "加载权值数据完成" << endl;
}

void test(){
    int cnt = 0;
    FILE * f_test_x;
    FILE * f_test_y;
    f_test_x = fopen("./tc/t10k-images.idx3-ubyte", "rb");
    f_test_y = fopen("./tc/t10k-labels.idx1-ubyte", "rb");
    unsigned char test_x[digit_vec],test_y[result_num];
    int useless[1000];
    fread(useless, 1, 16, f_test_x);
    fread(useless, 1, 8, f_test_y);
   
    int test_success_count = 0;
    while(!feof(f_test_x) && !feof(f_test_y)){
        memset(test_x, 0, 784);
        memset(test_y, 0, 10);
        fread(test_x, 1, 784, f_test_x);
        fread(test_y, 1, 1, f_test_y);
        for(int i = 0;i < 784; i ++){
            if((unsigned int)test_x[i] < 128){
                input[i] = 0;
            }
            else{
                input[i] = 1;
            }
        }
        for (int k = 0; k < 10; k++){
            target[k] = 0;
        } 
        int label_value = (unsigned int)test_y[0];
        target[label_value] = 1;
        op1();
        op2();
        double max_value = -99999;
        int max_index = 0;
        for (int k = 0; k < 10; k++){
            if (output2[k] > max_value){
                max_value = output2[k];
                max_index = k;
            }
        }

        cnt ++;
        //output == target
        if (target[max_index] == 1){
            test_success_count ++;
        }
        if (cnt % 1000 == 0){
            cout << "test num  success: " << test_success_count << endl;
        }
    }
    cout << endl;
    cout << "The success rate: " << test_success_count * 1.0 / cnt << endl;
    return;
}

int main(){
    init_model();
    test();
    return 0;
}
