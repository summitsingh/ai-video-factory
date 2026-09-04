import {Composition} from 'remotion';
import {SyntheticVideo} from './SyntheticVideo';
import {parseEditDocument, type EditDocument} from './schema';

const defaultEditDocument: EditDocument = {
  schema_version: 1,
  width: 1280,
  height: 720,
  fps: 30,
  duration_frames: 90,
  scenes: [
    {
      id: 'synthetic-title',
      from_frame: 0,
      duration_frames: 90,
      title: 'Synthetic video',
      caption: 'Local render',
    },
  ],
};

export const RemotionRoot = () => {
  return (
    <Composition
      component={SyntheticVideo}
      defaultProps={defaultEditDocument}
      durationInFrames={defaultEditDocument.duration_frames}
      fps={defaultEditDocument.fps}
      height={defaultEditDocument.height}
      id="SyntheticVideo"
      calculateMetadata={({props}) => {
        const edit = parseEditDocument(props);
        return {
          durationInFrames: edit.duration_frames,
          fps: edit.fps,
          height: edit.height,
          props: edit,
          width: edit.width,
        };
      }}
      width={defaultEditDocument.width}
    />
  );
};
