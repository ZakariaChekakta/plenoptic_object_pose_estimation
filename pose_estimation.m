%
%
% Project   Comparative Study of Filtering-based and AI-based Orbital Capturing/Docking Architectures for Debris Removal: Monocular and Depth Imaging Approaches     
% authors   Zakaria Chekakta <zakaria.chekakta@city.ac.uk>
% copyright Copyright (c) 2022, City University Of London, All rights reserved.
%
%
%% Load training data

%  Initialisation 
clc, clear;
close all;
rng('default'); rng(1);



num_traj=10;
%TM_cam={num_traj};
%TM_jason{1}=zeros(1,16);

TrainDataFilesPath = fullfile('C:\Users\sbrr512\OneDrive - City, University of London\Desktop\ESA_ADR_scenario','plenoptic_dataset');


[camera_pose_tr, rgb_im_tr,depth_image_tr ]=get_data(TrainDataFilesPath,num_traj);
initial_jason_pose=[-0.020751	-0.441248	-0.01571	-0.897008	341.192993	1232.60437	-1113.247681];
initial_jason_pose_r=initial_jason_pose(1:4);
initial_jason_pose_t=initial_jason_pose(5:7);

initial_jason_pose_m=quat2tform(initial_jason_pose_r);
initial_jason_pose_m(1:3,4)=initial_jason_pose_t;



for i =1:num_traj 
    pose_camera=[camera_pose_tr{i}(:,1) camera_pose_tr{i}(:,2) camera_pose_tr{i}(:,3) camera_pose_tr{i}(:,4) camera_pose_tr{i}(:,5) camera_pose_tr{i}(:,6) camera_pose_tr{i}(:,7)];
    
    for j =1:10
        CamPos_Rot=table2array(pose_camera(j,1:4));
        CamPos_T=table2array(pose_camera(j,5:7));
        TM_cam{i}{j}=quat2tform(CamPos_Rot);
        TM_cam{i}{j}(1:3,4)=CamPos_T';  
        
        TM_cam{i}{j}=TM_cam{i}{j}\initial_jason_pose_m;
        

    end
end

k=1;
for i =1:num_traj      
    traj_length=size(camera_pose_tr{i});
    for j=1:traj_length(1)
        im_tr(:,:,1:3,k)=rgb_im_tr{i}{j}(:,:,1:3);
        depth_im_tr(:,:,1,k)=depth_image_tr{i}{j}(:,:);

        
        %im_tr(:,:,4,k)=depth_im_tr(:,:,k);
        %edge_depth_im_tr(:,:,k)=edge(depth_im_tr(:,:,k));

        TM_tr(k,:)=TM_cam{i}{j}(:);
        k=k+1;
    end
end




image_tr=im_tr(:,:,:,1:100);
depth_image_training=depth_im_tr(:,:,1,1:100);
TM_train=TM_tr(1:100,:);

image_val=im_tr(:,:,:,61:80);
depth_image_val=depth_im_tr(:,:,1,61:80);
TM_val=TM_tr(61:80,:);

image_test=im_tr(:,:,:,81:100);
depth_image_test=depth_im_tr(:,:,1,81:100);

TM_test=TM_tr(81:100,:);



%%
% 
% clc
% layers = [
%     
%     imageInputLayer([size(im_tr,1) size(im_tr,2) size(im_tr,3)],'Name','Input',Normalization="none")
% 
%     convolution2dLayer(5,64,'Stride',2,'Padding',[3 3 3 3],'Name','Conv1')
%     reluLayer('Name','Relu1')
% 
%     fullyConnectedLayer(1024,'Name','fc1')
%     
%     %regressionLayer
%      softmaxLayer];
% 
% net = dlnetwork(layers);
%%
% 
% %%
% if canUseGPU
%     executionEnvironment = "gpu";
%     numberOfGPUs = gpuDeviceCount("available");
%     pool = parpool(numberOfGPUs);
% else
%     executionEnvironment = "cpu";
%     pool = parpool;
% end
% 
% %%
% 
% numWorkers = pool.NumWorkers;
% numEpochs = 20;
% miniBatchSize = 128;
% velocity = [];
% if executionEnvironment == "gpu"
%      miniBatchSize = miniBatchSize .* numWorkers
% end
% 
% workerMiniBatchSize = floor(miniBatchSize ./ repmat(numWorkers,1,numWorkers));
% remainder = miniBatchSize - sum(workerMiniBatchSize);
% workerMiniBatchSize = workerMiniBatchSize + [ones(1,remainder) zeros(1,numWorkers-remainder)]
% 
% batchNormLayers = arrayfun(@(l)isa(l,"nnet.cnn.layer.BatchNormalizationLayer"),net.Layers);
% batchNormLayersNames = string({net.Layers(batchNormLayers).Name});
% state = net.State;
% isBatchNormalizationStateMean = ismember(state.Layer,batchNormLayersNames) & state.Parameter == "TrainedMean";
% isBatchNormalizationStateVariance = ismember(state.Layer,batchNormLayersNames) & state.Parameter == "TrainedVariance";
% 
% 
% monitor = trainingProgressMonitor( Metrics="TrainingLoss", ...
%     Info=["Epoch" "Workers"], ...
%     XLabel="Iteration");
% '''
%%


%%
%^^^^^^^^^^^^^^^^^^^^^^^^^^ Trainig ^^^^^^^^^^^^^^
clc

TM_2=TM_train;
TM_2(:,17:1024)=0;
TM_val(:,17:1024)=0;
validationData=cell(1,2);
validationData{1,1}=image_val;
validationData{1,2}=TM_val;



doTraining = 0;
CNNlayers = [
    
    imageInputLayer([size(im_tr,1) size(im_tr,2) size(im_tr,3)],'Name','Input')

    convolution2dLayer(5,64,'Stride',2,'Padding',[3 3 3 3],'Name','Conv1')
    reluLayer('Name','Relu1')

    convolution2dLayer(5,64*2,'Stride',2,'Padding',[2 2 2 2],'Name','Conv2')
    reluLayer('Name','Relu2')

    convolution2dLayer(5,64*4,'Stride',2,'Padding',[2 2 2 2],'Name','Conv3')
    reluLayer('Name','Relu3')

    convolution2dLayer(5,64*8,'Stride',2,'Padding',[2 2 2 2],'Name','Conv4')
    reluLayer('Name','Relu4')

     convolution2dLayer(5,64*8,'Stride',2,'Padding',[2 2 2 2],'Name','Conv5')
    reluLayer('Name','Relu5')

    fullyConnectedLayer(1024,'Name','fc1')
    
    regressionLayer];



% Train CNN - net1


maxEpochs =300;
miniBatchSize = 10; %5
options = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.002, ...
    'GradientThreshold',1, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'ValidationFrequency',100 ,...
    'ValidationData',validationData, ...
    'ExecutionEnvironment', 'gpu',...
    'Verbose',1);

if doTraining
    CNN = trainNetwork(image_tr,TM_2,CNNlayers,options);
    save("cnn_net.mat","CNN");
else
    load("cnn_net.mat");
end


%%


% clc
% 
% doTraining = 1;
% 
% for i=1:100
%     depth_raw(i,:)= reshape(depth_image_training(:,:,1,i).',1,[]);
% end
% 
% depth_raw_tr=depth_raw(1:100,:);
% TM_train=TM_2(1:100,:);
% 
% depth_raw_test=depth_raw(81:100,:);
% 
% depth_raw_val=depth_raw(61:80,:);
% 
% validationData=cell(1,2);
% validationData{1,1}=depth_raw_val;
% validationData{1,2}=TM_val;
% 
% %TM_2=TM_tr;
% %TM_2(:,17:1024)=0;
% 
% numFeatures=size(depth_raw,2);
% maxEpochs =100;
% miniBatchSize = 1; %5
% 
% Dlayers = [   
%     %featureInputLayer(numFeatures)
%     sequenceInputLayer(numFeatures)
%     
%     lstmLayer(1000,'OutputMode','sequence','Name','DLSTM1')
% 
%     lstmLayer(1000,'OutputMode','sequence','Name','DLSTM2')
% 
%     lstmLayer(1000,'OutputMode','sequence','Name','DLSTM3')
%     %convolution1dLayer(11,3)
%     %reluLayer
%     %fullyConnectedLayer(1024,'Name','Dfc1')  
% 
%     %fullyConnectedLayer(1024,'Name','Dfc4')  
%     %fullyConnectedLayer(1024,'Name','Dfc5')  
% 
%     regressionLayer];
% 
% Doptions = trainingOptions('sgdm', ...
%     'MaxEpochs',maxEpochs, ...
%     'MiniBatchSize',miniBatchSize, ...
%     'InitialLearnRate',0.001, ...
%     'GradientThreshold',1, ...
%     'Shuffle','once', ...
%     'Plots','training-progress',...
%      'ValidationFrequency',10 ,...
%     'ValidationData',validationData, ...
%     'ExecutionEnvironment', 'gpu',...
%     'Verbose',1);
% 
% 
% 
% if doTraining
%     DNet = trainNetwork(depth_raw_tr,TM_2,Dlayers,Doptions);
%     save("DNet.mat","DNet");
% else
%     load("DNet.mat");
% end

%%
clc
doTraining=0;

 validationData=cell(1,2);
 validationData{1,1}=depth_image_val;
 validationData{1,2}=TM_val;

maxEpochs =100;
miniBatchSize = 1; %5
DCNNlayers = [   
    imageInputLayer([size(depth_image_training,1) size(depth_image_training,2) size(depth_image_training,3)],'Name','DInput')
   
    convolution2dLayer(5,64,'Stride',2,'Padding',[3 3 3 3],'Name','DConv1')
    reluLayer('Name','DRelu1')

    convolution2dLayer(5,64*2,'Stride',2,'Padding',[2 2 2 2],'Name','DConv2')
    reluLayer('Name','DRelu2')

    convolution2dLayer(5,64*4,'Stride',2,'Padding',[2 2 2 2],'Name','DConv3')
    reluLayer('Name','DRelu3')

    convolution2dLayer(5,64*8,'Stride',2,'Padding',[2 2 2 2],'Name','DConv4')
    reluLayer('Name','DRelu4')

    %convolution2dLayer(3,64*16,'Stride',2,'Padding',[2 2 2 2],'Name','DConv5')
    %reluLayer('Name','DRelu5')
    %fullyConnectedLayer(2048,'Name','Dfc0')  
    fullyConnectedLayer(1024,'Name','Dfc1')  
    regressionLayer];

Doptions = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.001, ...
    'GradientThreshold',1, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'ValidationFrequency',10 ,...
    'ValidationData',validationData, ...
    'ExecutionEnvironment', 'gpu',...
    'Verbose',1);


if doTraining
    DCNNet = trainNetwork(depth_image_training,TM_2,DCNNlayers,Doptions);
    save("DCNNet.mat","DCNNet");
else
    load("DCNNet.mat");
end

%%
clc
featureLayer = 'fc1';

trainingFeatures1 = activations(CNN, image_tr, featureLayer, ...
    'MiniBatchSize', 1, 'OutputAs', 'columns');

featureLayer1 = 'Dfc1';
trainingFeatures2 = activations(DCNNet, depth_image_training, featureLayer1, ...
    'MiniBatchSize', 1, 'OutputAs', 'columns');

trainingFeatures=[trainingFeatures1;trainingFeatures2];



valid_Features1 = activations(CNN, image_val, featureLayer, ...
    'MiniBatchSize', 1, 'OutputAs', 'columns');


valid_Features2 = activations(DCNNet, depth_image_val, featureLayer1, ...
    'MiniBatchSize', 1, 'OutputAs', 'columns');

valid_Features=[valid_Features1;valid_Features2];

%%

clc
doTraining=0;
miniBatchSize =4;
maxEpochs=1000;


validationData2=cell(1,2);
validationData2{1,1}=valid_Features;
validationData2{1,2}=TM_val';


options1 = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.1, ...
    'GradientThreshold',1, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'Verbose',true, ...
    'ValidationFrequency',10,...
    'ExecutionEnvironment', 'gpu',...
    'ValidationData',validationData2);

RNNlayers = [
    sequenceInputLayer(2048)

    lstmLayer(1000,'OutputMode','sequence','Name','LSTM1')

    lstmLayer(1000,'OutputMode','sequence','Name','LSTM2')

    lstmLayer(1000,'OutputMode','sequence','Name','LSTM3')
    
    fullyConnectedLayer(1024,'Name','rnn_fc1')

    regressionLayer];



if doTraining
    RNN = trainNetwork(trainingFeatures,TM_2',RNNlayers,options1);
    save("RNN.mat","RNN");
else
    load("RNN.mat");
end
%% 
% Test

clc
TM_test1=TM_test';
 %-------------- MAIN CODE ---------------
for i=1:20
   
    testFeatures1 = activations(CNN, image_test(:,:,:,i), featureLayer,'MiniBatchSize', 1, 'OutputAs', 'columns');
    testFeatures2 = activations(DCNNet, depth_image_test(:,:,1,i), featureLayer1,'MiniBatchSize', 1, 'OutputAs', 'columns');
    %testFeatures2 = activations(DNet, depth_raw_test(i,:), featureLayer1,'MiniBatchSize', 1, 'OutputAs', 'columns');
    
    testFeatures(:,i)=[testFeatures1;testFeatures2];
    prediction(:,i) = predict(RNN,testFeatures(:,i),'MiniBatchSize',1,'SequenceLength','longest');

     prediction1(:,i) = predict(CNN,image_test(:,:,:,i),'MiniBatchSize',1,'SequenceLength','longest');
     prediction2(:,i) = predict(DCNNet,depth_image_test(:,:,1,i),'MiniBatchSize',1,'SequenceLength','longest');
     %prediction2(:,i) = predict(DNet,depth_raw_test(i,:),'MiniBatchSize',1,'SequenceLength','longest');
   

    pred(:,:,i)=reshape(prediction(1:16,i),[4 4]);
    pred1(:,:,i)=reshape(prediction1(1:16,i),[4 4]);
    pred2(:,:,i)=reshape(prediction2(1:16,i),[4 4]);
    test_gt(:,:,i)=reshape(TM_test1(:,i),[4 4]);

   rot_data(i,:)=rotm2eul(pred(1:3,1:3,i))*180/pi;
   rot_data1(i,:)=rotm2eul(pred1(1:3,1:3,i))*180/pi;
   rot_data2(i,:)=rotm2eul(pred2(1:3,1:3,i))*180/pi;
   rot_gt_test(i,:)=rotm2eul(test_gt(1:3,1:3,i))*180/pi;
 

end

   
   
%TM_test1=TM_test';
   
  

%%


err1=test_gt(1:3,4,:)-pred1(1:3,4,:);
err2=test_gt(1:3,4,:)-pred2(1:3,4,:);

rerr1=rot_gt_test-rot_data1;
rerr2=rot_gt_test-rot_data2;

figure(); 
 subplot(3,1,1);
 x_err=err1(1,:);
 plot(x_err,'b.-');
 hold on
 plot(err2(1,:),'r.-');
grid on
  
subplot(3,1,2);
y_err=err1(2,:);
 plot(y_err,'b.-');
  hold on
 plot(err2(2,:),'r.-');
grid on
 subplot(3,1,3);
 z_err=err1(3,:);
 plot(z_err,'b.-');
  hold on
 plot(err2(3,:),'r.-');
grid on

figure(2); 
 subplot(3,1,1);
 plot(rerr1(:,1),'r.-');
 hold on
 plot(rerr2(:,1),'b.-');
grid on
  
subplot(3,1,2);
 plot(rerr1(:,2),'r.-');
 hold on
 plot(rerr2(:,2),'b.-');
grid on
 subplot(3,1,3);
 plot(rerr1(:,3),'r.-');
 hold on
 plot(rerr2(:,3),'b.-');
grid on

%%
clc

I=depth_image_test(:,:,1,7);
corners=detectHarrisFeatures(I);
[features, valid_corners ]= extractFeatures(I,corners);

figure(2)
imshow(I)
hold on
plot(valid_corners)

%%
%detector=yolov2ObjectDetector("darknet19-coco");
%testImage=imread(imaga(:,:,:,10));
%[bbox, scores, labele ]=detect(detector, testImage);


