import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
import shutil
import os
from subprocess import Popen, PIPE
import matplotlib
import imageio
import base64



class FFMpegWriter(object):
    '''A writer of video files from Matplotlib figures.

    This class uses FFMpeg as the basis for writing frames to a video file.

    Parameters
    ----------
    filename : string
        The filename of the generated video file.
    codec : string
        The codec for FFMpeg to use. If it is not given, a suitable codec
        will be guessed based on the file extension.
    framerate : integer
        The number of frames per second of the generated video file.
    quality : string
        The quality of the encoding for lossy codecs. Please refer to FFMpeg documentation.
    preset : string
        The preset for the quality of the encoding. Please refer to FFMpeg documentation.

    Raises
    ------
    ValueError
        If the codec was not given and could not be guessed based on the file extension.
    RuntimeError
        If something went wrong during initialization of the call to FFMpeg. Most likely,
        FFMpeg is not installed and/or not available from the commandline.
    '''
    def __init__(self, filename, codec=None, framerate=24, quality=None, preset=None, ffmpeg_path=None):
        if codec is None:
            extension = os.path.splitext(filename)[1]
            if extension == '.mp4' or extension == '.avi':
                codec = 'libx264'
            elif extension == '.webm':
                codec = 'libvpx-vp9'
            else:
                raise ValueError('No codec was given and it could not be guessed based on the file extension.')

        self.is_closed = True
        self.filename = filename
        self.codec = codec
        self.framerate = framerate

        if ffmpeg_path is None:
            ffmpeg_path = 'ffmpeg'

        if shutil.which(ffmpeg_path) is None:
            raise RuntimeError('ffmpeg was not found. Did you install it and is it accessible, either from PATH or from the HCIPy configuration file?')

        if codec == 'libx264':
            if quality is None:
                quality = 10

            if preset is None:
                preset = 'veryslow'

            command = [
                ffmpeg_path, '-y', '-nostats', '-v', 'quiet', '-f', 'image2pipe',
                '-vcodec', 'png', '-r', str(framerate), '-threads', '0', '-i', '-',
                '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', preset, '-r',
                str(framerate), '-crf', str(quality), filename
            ]
        elif codec == 'mpeg4':
            if quality is None:
                quality = 4

            command = [
                ffmpeg_path, '-y', '-nostats', '-v', 'quiet', '-f', 'image2pipe',
                '-vcodec', 'png', '-r', str(framerate), '-threads', '0', '-i', '-',
                '-vcodec', 'mpeg4', '-q:v', str(quality), '-r', str(framerate), filename
            ]
        elif codec == 'libvpx-vp9':
            if quality is None:
                quality = 30

            command = [
                ffmpeg_path, '-y', '-nostats', '-v', 'quiet', '-f', 'image2pipe',
                '-vcodec', 'png', '-r', str(framerate), '-threads', '0', '-i', '-'
            ]

            if quality < 0:
                command.extend(['-vcodec', 'libvpx-vp9', '-lossless', '1', '-r', str(framerate), filename])
            else:
                command.extend(['-vcodec', 'libvpx-vp9', '-crf', str(quality), '-b:v', '0', '-r', str(framerate), filename])
        else:
            raise ValueError('Codec unknown.')

        try:
            self.p = Popen(command, stdin=PIPE)
        except OSError:
            raise RuntimeError('Something went wrong when opening FFMpeg.')

        self.is_closed = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            import warnings
            warnings.warn('Something went wrong while closing FFMpeg...', RuntimeWarning)

    def add_frame(self, fig=None, data=None, cmap=None, dpi=None):
        '''Add a frame to the animation.

        Parameters
        ----------
        fig : Matplotlib figure
            The Matplotlib figure acting as the animation frame.
        data : ndarray
            The image data array acting as the animation frame.
        cmap : Matplotlib colormap
            The optional colormap for the image data.
        dpi : integer or None
            The number of dots per inch with which to save the matplotlib figure.
            If it is not given, the default Matplotlib dpi will be used.

        Raises
        ------
        RuntimeError
            If the function was called on a closed FFMpegWriter.
        '''
        if self.is_closed:
            raise RuntimeError('Attempted to add a frame to a closed FFMpegWriter.')

        if data is None:
            if fig is None:
                fig = matplotlib.pyplot.gcf()

            facecolor = list(fig.get_facecolor())
            facecolor[3] = 1

            fig.savefig(self.p.stdin, format='png', transparent=False, dpi=dpi, facecolor=facecolor)
        else:
            if cmap is not None:
                data = matplotlib.cm.get_cmap(cmap)(data, bytes=True)

            imageio.imwrite(self.p.stdin, data, format='png')

    def close(self):
        '''Close the animation writer and finish the video file.

        This closes the FFMpeg call.
        '''
        if not self.is_closed:
            self.p.stdin.close()
            self.p.wait()
            self.p = None
        self.is_closed = True

    def _repr_html_(self):
        '''Get an HTML representation of the generated video.

        Helper function for Jupyter notebooks. The video will be inline embedded in an
        HTML5 video tag using base64 encoding. This is not very efficient, so only use this
        for small video files.

        The FFMpegWriter must be closed for this function to work.

        Raises
        ------
        RuntimeError
            If the call was made on an open FFMpegWriter.
        '''
        if not self.is_closed:
            raise RuntimeError('Attempted to show the generated movie on an opened FFMpegWriter.')

        video = open(self.filename, 'rb').read()
        video = base64.b64encode(video).decode('ascii').rstrip()

        if self.filename.endswith('.mp4'):
            mimetype = 'video/mp4'
        elif self.filename.endswith('.webm'):
            mimetype = 'video/webm'
        elif self.filename.endswith('.avi'):
            mimetype = 'video/avi'
        else:
            raise RuntimeError('Mimetype could not be guessed.')

        output = '''<video controls><source src="data:{0};base64,{1}" type="{0}">Your browser does not support the video tag.</video>'''
        output = output.format(mimetype, video)

        return output
    

def make_xy_animation(name, fade=0.5, step=15, output='xy_evolution.mp4'):
    aei = fits.getdata(name + '/all_aei.fits')
    xyz = fits.getdata(name + '/all_xyz.fits')

    anim = FFMpegWriter(f'{name}/{output}.mp4', framerate=30)
    p0 = None

    for iyr in tqdm(range(0, aei.shape[0], step)):
        plt.clf()
        plt.figure(figsize=(6, 6))
        grid = plt.GridSpec(5, 1, wspace=0.5, hspace=1)
        # f, ax = plt.subplots(2, 1, figsize=(6, 8))
        t, m, r, x, y, z, vx, vy, vz = xyz[iyr, :, :].T
        plt.subplot(grid[:3, 0])
        # plt.scatter(inc/np.pi*180*np.cos(Omega), inc/np.pi*180*np.sin(Omega), s=m**0.5*1e3, c=a, alpha=0.8, vmin=45, vmax=150)
        plt.scatter(x, y, s=m**0.25*5e1, alpha=0.3, edgecolors='none')
        plt.xlim([-100, 100])
        plt.ylim([-100, 100])

        plt.ylabel('x')
        plt.xlabel('y')
        plt.title(f'time = {t[0]/1000:.1f} kyr; index = {iyr}')
        plt.gca().set_aspect('equal')
        t_p, m_p, r_p, x_p, y_p, z_p, vx_p, vy_p, vz_p = xyz[:, 0, :].T
        plt.plot(x_p[iyr-epoch:iyr], y_p[iyr-epoch:iyr], '-', linewidth=1, color='k')
        # t_p, m_p, r_p, a_p, e_p, inc_p, Omega_p, w_p, T_p, E_p, M_p = aei[:, 0, :].T
        # plt.plot(inc_p/np.pi*180*np.cos(Omega_p), inc_p/np.pi*180*np.sin(Omega_p), '-', linewidth=1, color='gray')
        plt.colorbar(label='semi-major-axis')

        plt.subplot(grid[3:, 0])
        
        plt.scatter(x, z, s=m**0.25*5e1, alpha=0.3, edgecolors='none')
        # print(np.any(np.isnan(x)))
        p = np.polyfit(x[~np.isnan(x)], z[~np.isnan(x)], deg=3)
        if p0 is None:
            p0 = p
        else:
            p0 = p0 * (1-fade) + p * fade

        xi = np.linspace(-80, 80, 100)
        plt.plot(xi, p0[0]*xi**3 + p0[1]*xi**2 + p0[2]*xi + p0[3], '--', color='C1')

        epoch = 100

        plt.plot(x_p[iyr-epoch:iyr], z_p[iyr-epoch:iyr], '-', linewidth=1, color='k')
        plt.xlabel('x (AU)')
        plt.ylabel('z (AU)')
        plt.xlim([-100, 100])
        plt.ylim([-30, 30])
        plt.gca().set_aspect('equal')
        # plt.tight_layout()
        # plt.axis('square')
        anim.add_frame()
        plt.close()

    
    anim.close()


def make_pq_animation(name, fade=0.5, step=15, output='pq_evolution.mp4'):
    aei = fits.getdata(name + '/all_aei.fits')
    xyz = fits.getdata(name + '/all_xyz.fits')


    anim = FFMpegWriter(f'{name}/{output}.mp4', framerate=30)
    p0 = None

    for iyr in tqdm(range(0, aei.shape[0], step)):
        plt.clf()
        plt.figure(figsize=(6, 6))
        grid = plt.GridSpec(5, 1, wspace=0.5, hspace=1)
        # f, ax = plt.subplots(2, 1, figsize=(6, 8))
        t, m, r, a, e, inc, Omega, w, T, E, M = aei[iyr, :, :].T
        plt.subplot(grid[:3, 0])
        plt.scatter(inc/np.pi*180*np.cos(Omega), inc/np.pi*180*np.sin(Omega), s=m**0.5*1e3, c=a, alpha=0.8, vmin=45, vmax=150)
        
        plt.xlim([-30, 30])
        plt.ylim([-30, 30])

        plt.ylabel('$i \sin(\Omega)$')
        plt.xlabel('$i \cos(\Omega)$')
        plt.title(f'time = {t[0]/1000:.1f} kyr; index = {iyr}')
        plt.gca().set_aspect('equal')

        
        t_p, m_p, r_p, a_p, e_p, inc_p, Omega_p, w_p, T_p, E_p, M_p = aei[:, 0, :].T
        plt.plot(inc_p/np.pi*180*np.cos(Omega_p), inc_p/np.pi*180*np.sin(Omega_p), '-', linewidth=1, color='gray')
        plt.colorbar(label='semi-major-axis')

        plt.subplot(grid[3:, 0])
        t, m, r, x, y, z, vx, vy, vz = xyz[iyr, :, :].T
        plt.scatter(x, z, s=m**0.25*5e1, alpha=0.3, edgecolors='none')
        # print(np.any(np.isnan(x)))
        p = np.polyfit(x[~np.isnan(x)], z[~np.isnan(x)], deg=3)
        if p0 is None:
            p0 = p
        else:
            p0 = p0 * (1-fade) + p * fade

        xi = np.linspace(-80, 80, 100)
        plt.plot(xi, p0[0]*xi**3 + p0[1]*xi**2 + p0[2]*xi + p0[3], '--', color='C1')

        t_p, m_p, r_p, x_p, y_p, z_p, vx_p, vy_p, vz_p = xyz[:, 0, :].T
        epoch = 100

        plt.plot(x_p[iyr-epoch:iyr], z_p[iyr-epoch:iyr], '-', linewidth=1, color='k')
        plt.xlabel('x (AU)')
        plt.ylabel('z (AU)')
        plt.xlim([-150, 150])
        plt.ylim([-30, 30])
        plt.gca().set_aspect('equal')
        # plt.tight_layout()
        # plt.axis('square')
        anim.add_frame()
        plt.close()

    
    anim.close()

    return anim

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='make vedio of Genga output')
    
    subparsers = parser.add_subparsers(help='vedio type')
    parser_pq = subparsers.add_parser('pq', help='paticle evolution in pq plane [icosO, isinO]')
    parser_pq.add_argument('name', type=str, help='name of the target folder')
    parser_pq.add_argument('--step', type=int, default=10, help='step of the animation')
    parser_pq.add_argument('-o', '--output', type=str, default='pq_evolution.mp4', help='output vedio name')

    def pq_vedio(args):
        make_pq_animation(args.name, step=args.step, output=args.output)

    parser_pq.set_defaults(func=pq_vedio)
    
    args = parser.parse_args()
    args.func(args)

    