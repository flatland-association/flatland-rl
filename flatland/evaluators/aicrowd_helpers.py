import os
import uuid
import subprocess
import glob
import random

###############################################################
# Expected Env Variables
###############################################################
# Default Values to be provided :
# AICROWD_IS_GRADING : true
# CROWDAI_IS_GRADING : true
# S3_BUCKET : aicrowd-production
# S3_UPLOAD_PATH_TEMPLATE : misc/flatland-rl-Media/{}.mp4
# AWS_ACCESS_KEY_ID
# AWS_SECRET_ACCESS_KEY
# http_proxy
# https_proxy
###############################################################
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", False)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", False)
S3_BUCKET = os.getenv("S3_BUCKET", "aicrowd-production")
S3_UPLOAD_PATH_TEMPLATE = os.getenv("S3_UPLOAD_PATH_TEMPLATE", "misc/flatland-rl-Media/{}.mp4")


def get_boto_client():
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise Exception("AWS Credentials not provided..")
    try:
        import boto3
    except ImportError as e:
        raise Exception(
                        "boto3 is not installed. Please manually install by : ",
                        " pip install -U boto3"
                        )

    return boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )


def is_aws_configured():
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        return False
    else:
        return True


def is_grading():
    return os.getenv("CROWDAI_IS_GRADING", False) or \
        os.getenv("AICROWD_IS_GRADING", False)


def upload_random_frame_to_s3(frames_folder):
    all_frames = glob.glob(os.path.join(frames_folder, "*.png"))
    random_frame = random.choice(all_frames)
    s3 = get_boto_client()
    if not S3_UPLOAD_PATH_TEMPLATE:
        raise Exception("S3_UPLOAD_PATH_TEMPLATE not provided...")
    if not S3_BUCKET:
        raise Exception("S3_BUCKET not provided...")
    
    image_target_key = S3_UPLOAD_PATH_TEMPLATE.replace(".mp4", ".png").format(str(uuid.uuid4()))
    s3.put_object(
        ACL="public-read",
        Bucket=S3_BUCKET,
        Key=image_target_key,
        Body=open(random_frame, 'rb')
    )
    return image_target_key


def upload_to_s3(localpath):
    s3 = get_boto_client()
    if not S3_UPLOAD_PATH_TEMPLATE:
        raise Exception("S3_UPLOAD_PATH_TEMPLATE not provided...")
    if not S3_BUCKET:
        raise Exception("S3_BUCKET not provided...")
    
    image_target_key = S3_UPLOAD_PATH_TEMPLATE.format(str(uuid.uuid4()))
    s3.put_object(
        ACL="public-read",
        Bucket=S3_BUCKET,
        Key=image_target_key,
        Body=open(localpath, 'rb')
    )
    return image_target_key


def make_subprocess_call(command, shell=False):
    result = subprocess.run(
                command.split(),
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
    stdout = result.stdout.decode('utf-8')
    stderr = result.stderr.decode('utf-8')
    return result.returncode, stdout, stderr


def generate_movie_from_frames(frames_folder):
    """
        Expects the frames in the  frames_folder folder 
        and then use ffmpeg to generate the video
        which writes the output to the frames_folder
    """
    # Generate Thumbnail Video
    print("Generating Thumbnail...")
    frames_path = os.path.join(frames_folder, "flatland_frame_%04d.png")
    thumb_output_path = os.path.join(frames_folder, "out_thumb.mp4")
    return_code, output, output_err = make_subprocess_call(
        "ffmpeg -r 7 -start_number 0 -i " + 
        frames_path + 
        " -c:v libx264 -vf fps=7 -pix_fmt yuv420p -s 320x320 " + 
        thumb_output_path
    )
    if return_code != 0:
        raise Exception(output_err)

    # Generate Normal Sized Video
    print("Generating Normal Video...")
    frames_path = os.path.join(frames_folder, "flatland_frame_%04d.png")
    output_path = os.path.join(frames_folder, "out.mp4")
    return_code, output, output_err = make_subprocess_call(
        "ffmpeg -r 7 -start_number 0 -i " + 
        frames_path + 
        " -c:v libx264 -vf fps=7 -pix_fmt yuv420p -s 600x600 " + 
        output_path
    )
    if return_code != 0:
        raise Exception(output_err)

    return output_path, thumb_output_path

