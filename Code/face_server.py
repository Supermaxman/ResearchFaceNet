import os
import time
import cv2
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify

import face_identity
import face_embedding
import align_dlib

app = Flask(__name__)

# cd "C:\Users\maw150130\Desktop\ResearchFaceNet\ResearchFaceNet"
# python3 face_detect.py --threshold 0.925 -cx 1280 -cy 720 -ux 1280 -uy 720 --ident_store_path identity-store.pkl -kw 6 -kf 26 --fullscreen
# --threshold 0.925 -cx 1280 -cy 720 -ux 1280 -uy 720 --ident_store_path identity-store.pkl -kw 6 -kf 26

face_img_size = 96
face_emb_size = 128
threshold = 0.925
emb_model_path = 'InceptionResNetV1Small-VGGFace2-v1-1000.pb'
ident_store_path = 'identity-store.pkl'

if os.path.isfile(ident_store_path):
    print('Loading identities...')
    ident_store = face_identity.IdentityStore.load(ident_store_path)
    print('Loaded {} identities.'.format(len(ident_store)))
else:
    ident_store = face_identity.IdentityStore(face_emb_size)

print('Loading embedding model...')
emb_model = face_embedding.EmbeddingModel(face_img_size, emb_model_path)
emb_model.load()

print('Warming up embeddings query...')
emb_model.create_embeddings(np.zeros([1, face_img_size, face_img_size, 3]))

print('Loading dlib face detector...')
dlib_model = align_dlib.AlignDlib('shape_predictor_68_face_landmarks.dat')
dlib_align = align_dlib.AlignDlib.INNER_EYES_AND_BOTTOM_LIP


def run_model(image):
    dlib_face_rects = dlib_model.getAllFaceBoundingBoxes(image, rgb=True)
    f_imgs = []
    f_rects = []
    for idx, rect in enumerate(dlib_face_rects):
        f_aligned = dlib_model.align(
            face_img_size,
            image,
            rect,
            landmarkIndices=dlib_align)
        f_imgs.append(f_aligned)
        f_rect = (int(rect.left()), int(rect.top()), int(rect.width()), int(rect.height()))
        f_rects.append(f_rect)
    f_embs = emb_model.create_embeddings(f_imgs)
    return f_imgs, f_rects, f_embs


def process_image(file):
    image_str = file.read()
    image = np.fromstring(image_str, np.uint8)
    # Read image as BGR
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    # convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def identify(f_rects, f_embs):
    idents = []
    for idx, f_emb in enumerate(f_embs):
        face_ident = ident_store.find_identity(f_emb, threshold)
        if face_ident is None:
            idents.append(None)
        else:
            ident_dict = face_ident.serialize()
            ident_dist = float(np.square(np.linalg.norm(face_ident.avg_embedding - f_emb)))
            ident_dict['distance'] = ident_dist
            ident_dict['rect'] = f_rects[idx]
            idents.append(ident_dict)
    return idents


# Query for identity with image from client
@app.route('/identity', methods=['GET', 'POST'])
def identity():
    start_time = time.time()
    userid = request.args.get('userid')
    if request.method == 'POST':
        print('[UserId {}] requesting identification...'.format(userid))
        if 'image' not in request.files:
            return 'No image provided by userid {}'.format(userid)
        file = request.files['image']
        image = process_image(file)
        print('[UserId {}] image is {}x{}...'.format(userid, image.shape[1], image.shape[0]))
        f_imgs, f_rects, f_embs = run_model(image)
        print('[UserId {}] image contains {} faces...'.format(userid, f_embs.shape[0]))
        idents = identify(f_rects, f_embs)
        print('[UserId {}] Completed in {:.4} seconds'.format(userid, time.time() - start_time))
        return jsonify(idents)
    else:
        print('[UserId {}] requesting identity list...'.format(userid))
        return jsonify([ident.serialize() for ident in ident_store.identities])


# Create an identity for a person
@app.route('/create', methods=['POST'])
def create():
    start_time = time.time()
    userid = request.args.get('userid')
    print('[UserId {}] creating identity...'.format(userid))
    if 'image' not in request.files:
        return 'No image provided by userid {}'.format(userid)
    ident_name = request.form['name']
    file = request.files['image']
    image = process_image(file)
    print('[UserId {}] image is {}x{}...'.format(userid, image.shape[1], image.shape[0]))
    f_imgs, f_rects, f_embs = run_model(image)
    print('[UserId {}] image contains {} faces...'.format(userid, f_embs.shape[0]))
    assert len(f_embs) == 1
    ident_store.add_identity(ident_name, f_embs[0])
    idents = identify(f_rects, f_embs)
    print('[UserId {}] Completed in {:.4} seconds'.format(userid, time.time() - start_time))
    return jsonify(idents)


@app.route('/addimage', methods=['POST'])
def addimage():
    start_time = time.time()
    userid = request.args.get('userid')
    print('[UserId {}] adding identity image...'.format(userid))
    if 'image' not in request.files:
        return 'No image provided by userid {}'.format(userid)
    ident_id = int(request.form['id'])
    file = request.files['image']
    image = process_image(file)
    print('[UserId {}] image is {}x{}...'.format(userid, image.shape[1], image.shape[0]))
    f_imgs, f_rects, f_embs = run_model(image)
    print('[UserId {}] image contains {} faces...'.format(userid, f_embs.shape[0]))
    assert len(f_embs) == 1
    ident_store.add_embedding(ident_id, f_embs[0])
    idents = identify(f_rects, f_embs)
    print('[UserId {}] Completed in {:.4} seconds'.format(userid, time.time() - start_time))
    return jsonify(idents)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
