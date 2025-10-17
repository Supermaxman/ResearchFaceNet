# MIT License
# 
# Copyright (c) 2018 Maxwell Weinzierl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import traceback
import argparse
import logging
import random
import os
import time
import ctypes
import multiprocessing
import threading

import numpy as np
import cv2
import tkinter as tk
import tkinter.font as tkfont
from PIL import Image
from PIL import ImageTk

import face_identity
import face_embedding
import align_dlib
import popup_keyboard
import face_utils

    
def face_start(dlib_model_path, emb_model_path,
               face_img_size, face_emb_size, cam_resolution, 
               threshold, max_nrof_faces, 
               shared_face_started, shared_face_ended,
               shared_new_image, shared_new_faces, shared_app_ended,
               shared_image_lock, shared_image, 
               shared_face_lock, shared_face_embs, 
               shared_face_rects, shared_nrof_faces):
    try:
        #face_thread_delay = 0.1
        use_rgb_for_detect=False
        
        print('Loading dlib model...')
        dlib_model = align_dlib.AlignDlib(dlib_model_path)

        print('Loading embedding model...')
        emb_model = face_embedding.EmbeddingModel(face_img_size, emb_model_path)
        emb_model.load()

        print('Warming up embeddings query...')
        emb_model.create_embeddings(np.zeros([1, face_img_size, face_img_size, 3]))
            
        image = np.frombuffer(shared_image, dtype=np.uint8)
        # HEIGHT, WIDTH, BGR.
        image = np.reshape(image, [cam_resolution[1], cam_resolution[0], 3])
    
        face_embs = np.frombuffer(shared_face_embs, dtype=float)
        face_embs = np.reshape(face_embs, [max_nrof_faces, face_emb_size])
        face_rects = np.frombuffer(shared_face_rects, dtype=int)
        face_rects = np.reshape(face_rects, [max_nrof_faces, 4])
        # HEIGHT, WIDTH, BGR.
        local_image = np.zeros([cam_resolution[1], cam_resolution[0], 3], dtype=np.uint8)

        print('Starting face detection...')
    except Exception as e:
        print('Exception loading face process!')
        traceback.print_exc()
        shared_app_ended.set()
    finally:
        shared_face_started.set()

    try:
        while not shared_app_ended.is_set():
            # wait until there is a new image to process...
            shared_new_image.wait()
            start_time = time.time()
            # this image could have changed since the event, but who cares, it would only be newer.
            with shared_image_lock:
                # read in a new image
                np.copyto(local_image, image)
                # clear new image flag since we have now read this image
                shared_new_image.clear()

            # process local image to find the face rectangles and embeddings
            # HEIGHT, WIDTH.
            cv_gray_img = cv2.cvtColor(local_image, cv2.COLOR_BGR2GRAY)
            # use gray image for face detection for speed concerns
            dlib_face_rects = dlib_model.getAllFaceBoundingBoxes(cv_gray_img, rgb=False)
            nrof_found_faces = len(dlib_face_rects)
        
            local_face_embs = np.zeros([max_nrof_faces, face_emb_size], dtype=float)
            local_face_rects = np.zeros([max_nrof_faces, 4], dtype=int)
            if nrof_found_faces > 0:
                if nrof_found_faces > max_nrof_faces:
                    raise Exception('More than max number of faces detected!')
                # create a RGB color image for face cropping.
                # HEIGHT, WIDTH, RGB.
                cv_color_img = cv2.cvtColor(local_image, cv2.COLOR_BGR2RGB)
                # nrof_found_faces, face_img_size, face_img_size, RGB
                f_imgs = np.zeros([nrof_found_faces, face_img_size, face_img_size, 3], dtype=np.uint8)
                for idx, rect in enumerate(dlib_face_rects):
                    f_aligned = dlib_model.align(face_img_size, cv_color_img, rect, landmarkIndices=align_dlib.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
                    f_imgs[idx, :, :, :] = f_aligned
                    local_face_rects[idx, :] = [int(rect.left()), int(rect.top()), int(rect.width()), int(rect.height())]
                
                local_face_embs[:nrof_found_faces, :] = emb_model.create_embeddings(f_imgs)

            with shared_face_lock:
                shared_nrof_faces.value = nrof_found_faces
                np.copyto(face_embs[:nrof_found_faces, :], local_face_embs[:nrof_found_faces, :])
                np.copyto(face_rects[:nrof_found_faces, :], local_face_rects[:nrof_found_faces, :])
                shared_new_faces.set()
            
            time_spent = time.time() - start_time   
            #print('Found {} faces in {:.2f}!'.format(nrof_found_faces, time_spent))
        
            # if we spent less time then our goal sleep for the rest
            #TODO determine if there's any reason to sleep since we are now using events. 
            #TODO why not just always try to process images? is it killing the cpu? test this
            #if time_spent < face_thread_delay:
            #    time.sleep(face_thread_delay - time_spent)
            ## otherwise we are not able to make our goal, so print it and do not sleep.
            #else:
            #    print('Not making time goal of {}!'.format(face_thread_delay))

    except Exception as e:
        print('Exception during face processing!')
        traceback.print_exc()
        shared_app_ended.set()
    finally:
        shared_face_ended.set()
        print('Face process ended.')
    

class FaceApp(object):
    def __init__(self, face_img_size, face_emb_size, cam_resolution, ui_resolution, 
                 keyboard_font_size, keyboard_keysize,
                 threshold, max_nrof_faces, fullscreen, raspberrypi, 
                 ident_store_path, dlib_model_path, emb_model_path):
        
        self.ident_store_path = ident_store_path
        self.dlib_model_path = dlib_model_path
        self.emb_model_path = emb_model_path

        # identity store
        if os.path.isfile(self.ident_store_path):
            print('Loading identities...')
            self.ident_store = face_identity.IdentityStore.load(self.ident_store_path)
            print('Loaded {} identities.'.format(len(self.ident_store)))
        else:
            self.ident_store = face_identity.IdentityStore(face_emb_size)


        # configuration
        self.face_img_size = face_img_size
        self.face_emb_size = face_emb_size
        self.cam_resolution = cam_resolution
        self.ui_resolution = ui_resolution
        self.threshold = threshold
        self.max_nrof_faces = max_nrof_faces
        self.fullscreen = fullscreen
        self.raspberrypi = raspberrypi
        self.ident_ui_image_scale = 1.5
        self.keyboard_keysize = keyboard_keysize
        self.keyboard_font_size = keyboard_font_size
        # end configuration

        # setup process shared objects
        self.shared_image_lock = multiprocessing.Lock()
        self.shared_face_lock = multiprocessing.Lock()

        self.shared_image = multiprocessing.RawArray(ctypes.c_uint8, self.cam_resolution[1] * self.cam_resolution[0] * 3)
        self.shared_face_embs = multiprocessing.RawArray(ctypes.c_double, self.max_nrof_faces * self.face_emb_size)
        self.shared_face_rects = multiprocessing.RawArray(ctypes.c_int, self.max_nrof_faces * 4)
        self.shared_nrof_faces = multiprocessing.RawValue(ctypes.c_int, 0)

        self.shared_face_started = multiprocessing.Event()
        self.shared_video_started = multiprocessing.Event()
        self.shared_app_started = multiprocessing.Event()
        self.shared_new_image = multiprocessing.Event()
        self.shared_new_faces = multiprocessing.Event()
        self.shared_face_ended = multiprocessing.Event()
        self.shared_video_ended = multiprocessing.Event()
        self.shared_app_ended = multiprocessing.Event()
        # end

        # setup tk GUI 
        self.close_lock = threading.Lock()
        root = tk.Tk()
        self.root = root
        self.panel = None

        root.title('ResearchFaceNet')
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        
        self.window_w, self.window_h = root.winfo_screenwidth(), root.winfo_screenheight()
        if self.fullscreen:
            root.attributes('-fullscreen', True) 
            
        self.font = tkfont.Font(family='Haettenschweiler', size=self.keyboard_font_size)

        loading_image = Image.open(os.path.join(os.path.dirname(__file__), 'Loading.png'))
        loading_image.thumbnail(self.ui_resolution, Image.ANTIALIAS)

        loading_image = ImageTk.PhotoImage(loading_image) 
                
        self.panel = tk.Label(root, image=loading_image)
        self.panel.image = loading_image
        # 5 columns, 4 rows total
        # row 0 to row 2, col 0 to col 3
        btn_sticky = tk.N + tk.S + tk.E + tk.W
        final_row = 6
        final_col = 3
        self.panel.grid(row=0, column=0, columnspan=4, rowspan=6, padx=5, pady=5, sticky=tk.W + tk.E + tk.N + tk.S)
        
        new_btn = tk.Button(root, text='Create Identity', command=self.add_identity, height=5, font=self.font)
        new_btn.grid(row=final_row, column=0, padx=5, pady=5, sticky=btn_sticky)
        
        add_btn = tk.Button(root, text='Add Image', command=self.add_embedding, height=5, font=self.font)
        add_btn.grid(row=final_row, column=1, padx=5, pady=5, sticky=btn_sticky)

        save_btn = tk.Button(root, text='Save', command=self.save, height=5, font=self.font)
        save_btn.grid(row=final_row, column=2, padx=5, pady=5, sticky=btn_sticky)

        exit_btn = tk.Button(root, text='Exit', command=self.on_close, height=5, font=self.font)
        exit_btn.grid(row=final_row, column=3, padx=5, pady=5, sticky=btn_sticky)

        for i in range(final_col + 1):
            root.grid_columnconfigure(i, weight=1)

        for i in range(final_row + 1):
            root.grid_rowconfigure(i, weight=1)

        # end

    def get_identity_ui_info(self):
        with self.shared_face_lock:
            nrof_faces = self.shared_nrof_faces.value
            local_image = np.copy(self.image)
            local_face_embs = np.copy(self.face_embs[:nrof_faces, :])
            local_face_rects = np.copy(self.face_rects[:nrof_faces, :])
        if nrof_faces == 0:
            #TODO error message, no identity found in image
            return None, None

        # take the largest face bounding box and use that one
        # TODO maybe go through all images 
        # (make this function a generator that yields each face?)
        max_ident_idx = face_utils.get_max_area_ident(local_face_rects, nrof_faces)

        ident_emb = local_face_embs[max_ident_idx]
        ident_rect = local_face_rects[max_ident_idx]
        (x, y, w, h) = face_utils.adjust_bounding_box(
            ident_rect[0], ident_rect[1], ident_rect[2], ident_rect[3], local_image.shape[1], local_image.shape[0])
        ident_ui_image = Image.fromarray(cv2.cvtColor(local_image[y:y+h, x:x+w, :], cv2.COLOR_BGR2RGB))
        
        ident_image_ui_resolution = (int(self.face_img_size * self.ident_ui_image_scale), int(self.face_img_size * self.ident_ui_image_scale))

        ident_ui_image.thumbnail(ident_image_ui_resolution, Image.ANTIALIAS)

        try:
            ident_ui_image = ImageTk.PhotoImage(ident_ui_image)
        except RuntimeError as e:
            print(str(e))
            return None, None

        return ident_emb, ident_ui_image

    def add_identity(self):
        # find the identity embeddings and images to use for creation
        new_ident_emb, new_ident_image = self.get_identity_ui_info()

        if new_ident_emb is None:
            return

        top_level = tk.Toplevel()
        
        top_level.title('Create Identity')

        if self.fullscreen:
            top_level.attributes('-fullscreen', True)

        label = tk.Label(top_level, text='Create Identity', font=self.font)
        label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.N + tk.S + tk.W)

        keyboard_entry = popup_keyboard.KeyboardEntry(top_level, keysize=self.keyboard_keysize, keycolor='white', font=self.font)
        keyboard_entry.grid(row=1, column=0, padx=5, pady=5, sticky=tk.S + tk.W)
        
        add_btn = tk.Button(top_level, text='Create', command=lambda: self.add_identity_final(new_ident_emb, keyboard_entry, top_level), font=self.font)
        add_btn.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky=tk.N + tk.S + tk.W + tk.E)
        
        cancel_btn = tk.Button(top_level, text='Cancel', command=lambda: self.add_identity_close(keyboard_entry, top_level), font=self.font)
        cancel_btn.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky=tk.N + tk.S + tk.W + tk.E)
        
        panel = tk.Label(top_level, image=new_ident_image)
        panel.image = new_ident_image
        panel.grid(row=0, column=3, columnspan=2, rowspan=2, padx=5, pady=5, sticky=tk.N + tk.S + tk.W + tk.E)
        
        top_level.protocol('WM_DELETE_WINDOW', lambda: self.add_identity_close(keyboard_entry, top_level))
        self.root.withdraw()


    def add_identity_final(self, new_ident_emb, keyboard_entry, top_level):
        new_ident_name = keyboard_entry.get()
        new_ident = self.ident_store.add_identity(new_ident_name, new_ident_emb)
        print('Created identity: {}'.format(new_ident))
        self.add_identity_close(keyboard_entry, top_level)


    def add_identity_close(self, keyboard_entry, top_level):
        # final close
        self.popup_close(top_level)
        # close popup ui
        keyboard_entry.destroy()
        
    def popup_close(self, top_level):
        self.root.deiconify()
        top_level.withdraw()
        top_level.destroy()

    def add_embedding(self):
        # find the identity embeddings and images to use for creation
        new_ident_emb, new_ident_image = self.get_identity_ui_info()

        if new_ident_emb is None:
            return

        # TODO add a list of identities or some searchable system.
        # TODO I need to be able to add embeddings for identities which do not fall within the threshold.
        found_ident = self.ident_store.find_identity(new_ident_emb, self.threshold)
        self.ident_store.add_embedding(found_ident.id, new_ident_emb)
        print('Added embedding to identity: {}'.format(found_ident))

    def save(self):
        print('Saving IdentityStore...')
        self.ident_store.save(self.ident_store_path)
        print('Saved IdentityStore.')

    def start(self):
        # create video stream process
        self.v_thread = threading.Thread(target=self.video_start, args=())
        self.v_thread.start()
        # end 

        # create face detection and recognition process
        self.f_process = multiprocessing.Process(
            target=face_start, # face_start_test
            args=(self.dlib_model_path, self.emb_model_path,
                  self.face_img_size, self.face_emb_size, self.cam_resolution, 
                  self.threshold, self.max_nrof_faces, 
                  self.shared_face_started, self.shared_face_ended,
                  self.shared_new_image, self.shared_new_faces, self.shared_app_ended,
                  self.shared_image_lock, self.shared_image, 
                  self.shared_face_lock, self.shared_face_embs, 
                  self.shared_face_rects, self.shared_nrof_faces))
        self.f_process.start()
        # end 
        
        # start ui loop
        self.root.mainloop()


    def video_start(self):
        try:
            self.image = np.frombuffer(self.shared_image, dtype=np.uint8)
            # HEIGHT, WIDTH, BGR.
            self.image = np.reshape(self.image, [self.cam_resolution[1], self.cam_resolution[0], 3])
            self.face_embs = np.frombuffer(self.shared_face_embs, dtype=float)
            self.face_embs = np.reshape(self.face_embs, [self.max_nrof_faces, self.face_emb_size])
            self.face_rects = np.frombuffer(self.shared_face_rects, dtype=int)
            self.face_rects = np.reshape(self.face_rects, [self.max_nrof_faces, 4])
            
            if self.raspberrypi:
                from pivideostream import PiVideoStream
                pi_fps = 32
                self.stream = PiVideoStream(resolution=self.cam_resolution, framerate=pi_fps)
                self.stream.start()
            else:
                from webcamvideostream import WebcamVideoStream
                cv_src = 0
                cv_fps = 60
                cv_stream = cv2.VideoCapture(cv_src)
                cv_stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_resolution[0])
                cv_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_resolution[1])
                cv_stream.set(cv2.CAP_PROP_FPS, cv_fps)
                self.stream = WebcamVideoStream(cv_stream)
                self.stream.start()

            print('Camera Settings: {} @ res {}'.format(self.image.shape, self.cam_resolution))

            local_image = np.zeros([self.cam_resolution[1], self.cam_resolution[0], 3], dtype=np.uint8)
            local_face_embs = np.zeros([self.max_nrof_faces, self.face_emb_size], dtype=float)
            local_face_rects = np.zeros([self.max_nrof_faces, 4], dtype=int)
            local_face_idents = np.arange(self.max_nrof_faces, dtype=int)
            ident_colors = dict()

            nrof_faces = 0
    
            start_time = time.time()
            fps_interval = 1.0
            fps_counter = 0
            fps = 0.0
            # allow the camera to warmup
            time.sleep(2.0)

            # wait for face process to load
            self.shared_face_started.wait()

        except Exception as e:
            print('Exception loading video thread!')
            traceback.print_exc()
            self.shared_app_ended.set()
        finally:
            self.shared_video_started.set()
            print('Starting video stream...')

        try:
            # capture frames from the camera
            while not self.shared_app_ended.is_set():
                # image comes from either CV2 or from Picamera as HEIGHT, WIDTH, BGR.
                stream_image = self.stream.read()
                np.copyto(local_image, stream_image)
                with self.shared_image_lock:
                    np.copyto(self.image, stream_image)
                    self.shared_new_image.set()
        
                with self.shared_face_lock:
                    local_new_faces = self.shared_new_faces.is_set()
                    if local_new_faces:
                        nrof_faces = self.shared_nrof_faces.value
                        np.copyto(local_face_embs[:nrof_faces, :], self.face_embs[:nrof_faces, :])
                        np.copyto(local_face_rects[:nrof_faces, :], self.face_rects[:nrof_faces, :])
                        # we've seen the faces so they are no longer new
                        self.shared_new_faces.clear()
                  
                # if we found new faces then recalculate the same identities
                if local_new_faces:
                    # go through all pairs of images and combine similar identities.
                    next_ident = self.ident_store.get_next_id()
                    local_face_idents[:nrof_faces] = np.arange(next_ident, next_ident + nrof_faces, dtype=int)
                    face_is_found = np.full([nrof_faces], False, dtype=bool)

                    unknown_face_idxs = []
                    known_face_idxs = []
                    found_identity_lookup = dict()
                    # find faces which are known
                    for a_idx in range(nrof_faces):
                        # consider voting, early exit, k-nn, or alternatives
                        known_ident = self.ident_store.find_identity(local_face_embs[a_idx], self.threshold)
                        if known_ident != None:
                            found_identity_lookup[a_idx] = known_ident
                            local_face_idents[a_idx] = known_ident.id
                            known_face_idxs.append(a_idx)
                            face_is_found[a_idx] = True
                        else:
                            unknown_face_idxs.append(a_idx)

                    final_unknown_face_idxs = unknown_face_idxs
                    #TODO determine if I even want to compare found images of an identity with previously unknown images.
                    #TODO on the surface it makes sense, but consider that any unknown images by now failed being classified by 
                    #TODO stored embeddings. You are essentially increasing the threshold by not strictly enforcing the threshold directly,
                    #TODO but only indirectly.
                    #final_unknown_face_idxs = []
                    ## attempt to label unknown faces with found known faces
                    #for u_idx in unknown_face_idxs:
                    #    k_found = False
                    #    for k_idx in known_face_idxs:
                    #        u_k_dist = np.linalg.norm(local_face_embs[u_idx] - local_face_embs[k_idx])**2
                    #        if u_k_dist < self.threshold:
                    #            found_identity_lookup[u_idx] = found_identity_lookup[k_idx]
                    #            local_face_idents[u_idx] = local_face_idents[k_idx]
                    #            known_face_idxs.append(u_idx)
                    #            face_is_found[u_idx] = True
                    #            # consider voting alternative to early exit
                    #            k_found = True
                    #            break
                    #    if not k_found:
                    #        final_unknown_face_idxs.append(u_idx)

                    # go through all the remaining unknown faces 
                    # and compare pairs for the same unknown faces
                    for u_idx_idx in range(len(final_unknown_face_idxs) - 1):
                        for c_idx_idx in range(1, len(final_unknown_face_idxs)):
                            u_idx = final_unknown_face_idxs[u_idx_idx]
                            c_idx = final_unknown_face_idxs[c_idx_idx]
                            c_u_dist = np.linalg.norm(local_face_embs[c_idx] - local_face_embs[u_idx])**2
                            if c_u_dist < self.threshold:
                                local_face_idents[c_idx] = local_face_idents[u_idx]


                # we draw the face bounding boxes every frame, 
                # but we only change them every time there is a new_faces=True
                for idx in range(nrof_faces):
                    rect = local_face_rects[idx]
                    ident = local_face_idents[idx]
                    is_found = face_is_found[idx]
                    if is_found:
                        found_ident = found_identity_lookup[idx]
                        display_str = found_ident.name
                        color = found_ident.color
                    else:
                        if ident not in ident_colors:
                            ident_colors[ident] = face_identity.generate_new_color(ident_colors.values(), pastel_factor=0.2)
                        display_str = 'Unknown {}'.format(ident)
                        color = ident_colors[ident]

                    (x, y, w, h) = (rect[0], rect[1], rect[2], rect[3])
                    cv2.rectangle(local_image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(local_image, display_str, (x + 10, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

                fps_counter += 1
                if (time.time() - start_time) > fps_interval:
                    fps = fps_counter / (time.time() - start_time)
                    fps_counter = 0
                    start_time = time.time()
                
                cv2.putText(local_image, 'FPS: {:.2f}'.format(fps), (10, local_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1)
                
                ui_image = Image.fromarray(cv2.cvtColor(local_image, cv2.COLOR_BGR2RGB))
                ui_image.thumbnail(self.ui_resolution, Image.ANTIALIAS)
                
                try:
                    # this step causes lots of issues when we try to exit the program,
                    # since the UI thread essentially must respond for this to complete.
                    # this means both the UI thread and this thread must be free to execute during
                    # a shutdown, which forces us to create a shutdown thread.
                    if self.shared_app_ended.is_set():
                        break
                    with self.close_lock:
                        if self.shared_app_ended.is_set():
                            break
                        ui_image = ImageTk.PhotoImage(ui_image)
                        self.panel.configure(image=ui_image)
                        self.panel.image = ui_image

                except RuntimeError as e:
                    print(str(e))

        except Exception as e:
            print('Exception during video processing!')
            traceback.print_exc()
            self.shared_app_ended.set()
        finally:
            try:
                self.stream.stop()
            except Exception as e:
                print('Exception video close!')
                traceback.print_exc()
            finally:
                self.shared_video_ended.set()
                print('Video thread ended.')

    def on_close_start(self):
        print('Ending app...')
        self.f_process.terminate()
        os._exit(0)
        self.root.withdraw()
        with self.close_lock:
            self.shared_app_ended.set()
        print('Waiting for video thread to end...')
        self.shared_video_ended.wait()
        self.v_thread.join()
        if self.v_thread.is_alive():
            print('Unable to exit video thread.')
        print('Waiting for face process to end...')
        self.shared_face_ended.wait()
        self.f_process.join()
        if self.f_process.is_alive():
            self.f_process.terminate()
        print('Ending main loop...')
        self.root.destroy()
        self.root.quit()
        print('Ended')
        # just to be sure everything closes
        os._exit(0)

    def on_close(self):
        # this is necessary to allow the ui thread to continue processing
        self.c_thread = threading.Thread(target=self.on_close_start, args=())
        self.c_thread.start()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run real time face detection and recognition using trained research model.')
    parser.add_argument('-is', '--face_img_size', default=96, type=int)
    parser.add_argument('-es', '--face_emb_size', default=128, type=int)
    parser.add_argument('-cx', '--cam_x_res', default=640, type=int)
    parser.add_argument('-cy', '--cam_y_res', default=480, type=int)
    parser.add_argument('-ux', '--ui_x_res', default=512, type=int)
    parser.add_argument('-uy', '--ui_y_res', default=384, type=int)
    parser.add_argument('-kf', '--keyboard_font_size', default=16, type=int)
    parser.add_argument('-kw', '--keyboard_keysize', default=4, type=int)
    # .982 accuracy             -> 1.225 threshold
    # .888 val / 1.12e-3 far    -> 0.925 threshold
    # consider even stricter threshold
    parser.add_argument('-t', '--threshold', default=0.925, type=float)
    parser.add_argument('-mf', '--max_faces', default=20, type=int)
    parser.add_argument('-fs', '--fullscreen', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('-isp', '--ident_store_path', default='identity-store-old.pkl', type=str)
    parser.add_argument('-dmp', '--dlib_model_path', default='shape_predictor_68_face_landmarks.dat', type=str)
    parser.add_argument('-emp', '--emb_model_path', default='InceptionResNetV1Small-VGGFace2-v1-1000.pb', type=str)
    parser.add_argument('-pi', '--raspberrypi', type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()

    try:
        face_app = FaceApp(args.face_img_size, args.face_emb_size, 
                           (args.cam_x_res, args.cam_y_res), 
                           (args.ui_x_res, args.ui_y_res), 
                           args.keyboard_font_size, args.keyboard_keysize,
                           args.threshold, args.max_faces, args.fullscreen, args.raspberrypi, 
                           args.ident_store_path, args.dlib_model_path, args.emb_model_path)
        face_app.start()
    except Exception as e:
        print(str(e))
        traceback.print_exc()
        time.sleep(10.0)
        