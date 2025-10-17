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


def get_max_area_ident(face_rects, nrof_faces):
    max_ident_idx = None
    max_ident_area = 0
    for ident_idx in range(nrof_faces):
        ident_rect = face_rects[ident_idx]
        # area = width * height
        ident_area = ident_rect[2] * ident_rect[3]
        if ident_area > max_ident_area:
            max_ident_idx = ident_idx
            max_ident_area = ident_area

    return max_ident_idx


def adjust_bounding_box(x, y, w, h, max_x, max_y):
    # if x or y are negative then we are going to 
    # crop from the edge of the image.
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    # if bounding box goes past limits then 
    # set limits as edge of box
    if x + w > max_x:
        w = max_x - x

    if y + h > max_y:
        h = max_y - y

    return x, y, w, h

