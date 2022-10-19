function [camera_pose rgb_image depth_image]=get_data(path,num_traj)


pose_gt={num_traj};


for i =1:num_traj         
        n_strPadded = sprintf( '%d', i) ;
        DataFileName = strcat(path,'\illum_trajs_poses');
        pose_path=strcat(DataFileName, '\traj', n_strPadded,'.xlsx');
        pose_gt{i}= readtable(pose_path);
        %pose_camera=[pose_gt{i}(:,1) pose_gt{i}(:,2) pose_gt{i}(:,3) pose_gt{i}(:,4) pose_gt{i}(:,5) pose_gt{i}(:,6) pose_gt{i}(:,7)];
end

for i =1:num_traj         
        n_strPadded = sprintf( '%d', i) ;
        DataFileName = strcat(path,'\light_field_frames');
        image_path=strcat(DataFileName, '\traj', n_strPadded);

        for j=1:10
            n_strPadded1 = sprintf( '%d', j) ;
            rgb_image_path=strcat(image_path, '\p', n_strPadded1,'.tiff');
            rgb_image{i}{j}=imresize(imread(rgb_image_path),0.2);

            depth_image_path=strcat(image_path, '\dp', n_strPadded1,'.png');
            %depth_image{i}{j}=imresize(rescale(imread(depth_image_path)),0.2);
            depth_image{i}{j}=imresize(imread(depth_image_path),0.1);
        end
       
end



camera_pose=pose_gt;
rgb_image=rgb_image;
depth_image=depth_image;
end