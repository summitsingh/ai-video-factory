import {AbsoluteFill, Sequence, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';
import type {EditDocument, EditScene} from './schema';

const SceneCard = ({scene}: {scene: EditScene}) => {
  const frame = useCurrentFrame();
  const {height} = useVideoConfig();
  const fadeFrames = Math.min(12, scene.duration_frames);

  return (
    <AbsoluteFill
      style={{
        alignItems: 'center',
        color: '#f8fafc',
        display: 'flex',
        fontFamily: 'Arial, sans-serif',
        justifyContent: 'center',
        opacity: interpolate(frame, [0, fadeFrames], [0, 1], {
          extrapolateLeft: 'clamp',
          extrapolateRight: 'clamp',
        }),
        padding: '10%',
        textAlign: 'center',
      }}
    >
      <div style={{maxWidth: '80%'}}>
        <div
          style={{
            fontSize: Math.max(48, Math.round(height * 0.105)),
            fontWeight: 700,
            letterSpacing: '-0.03em',
            lineHeight: 1.05,
          }}
        >
          {scene.title}
        </div>
        <div
          style={{
            backgroundColor: '#22d3ee',
            height: Math.max(6, Math.round(height * 0.01)),
            margin: `${Math.round(height * 0.035)}px auto 0`,
            width: '18%',
          }}
        />
      </div>
      <div
        style={{
          bottom: '10%',
          fontSize: Math.max(24, Math.round(height * 0.043)),
          left: '10%',
          letterSpacing: '0.02em',
          lineHeight: 1.25,
          position: 'absolute',
          right: '10%',
          color: '#94a3b8',
        }}
      >
        {scene.caption}
        {scene.visual && (
          <div style={{fontSize: Math.max(14, Math.round(height * 0.02)), marginTop: '4px'}}>
            Visual: {scene.visual}
          </div>
        )}
      </div>
    </AbsoluteFill>
  );
};

export const SyntheticVideo = ({scenes}: EditDocument) => {
  return (
    <AbsoluteFill style={{background: 'linear-gradient(135deg, #071a3d 0%, #000000 100%)'}}>
      {scenes.map((scene) => (
        <Sequence
          durationInFrames={scene.duration_frames}
          from={scene.from_frame}
          key={scene.id}
          layout="none"
        >
          <SceneCard scene={scene} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};