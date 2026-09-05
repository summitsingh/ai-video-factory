import {
  AbsoluteFill,
  Sequence,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import type {EditDocument, EditScene} from './schema';

const BACKGROUNDS = [
  'linear-gradient(135deg, #071a3d 0%, #000000 100%)',
  'linear-gradient(135deg, #3d071a 0%, #000000 100%)',
  'linear-gradient(135deg, #073d2a 0%, #000000 100%)',
];

const SceneCard = ({
  scene,
  index,
  total,
  isLast,
  sources,
}: {
  scene: EditScene;
  index: number;
  total: number;
  isLast: boolean;
  sources?: string[];
}) => {
  const frame = useCurrentFrame();
  const {height} = useVideoConfig();
  const fadeIn = Math.min(15, scene.duration_frames);
  const fadeOut = Math.min(8, scene.duration_frames);

  const enter = interpolate(frame, [0, fadeIn], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const exit = interpolate(
    frame,
    [scene.duration_frames - fadeOut, scene.duration_frames],
    [1, 0],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );
  const rise = interpolate(frame, [0, fadeIn], [36, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const captionIn = interpolate(frame, [fadeIn, fadeIn + 12], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        background: BACKGROUNDS[index % BACKGROUNDS.length],
        opacity: Math.min(enter, exit),
      }}
    >
      <AbsoluteFill
        style={{
          alignItems: 'center',
          display: 'flex',
          flexDirection: 'column',
          fontFamily: 'Arial, sans-serif',
          justifyContent: 'center',
          padding: '8% 10%',
          textAlign: 'center',
          transform: `translateY(${rise}px)`,
        }}
      >
        <div
          style={{
            color: '#22d3ee',
            fontSize: Math.max(20, Math.round(height * 0.032)),
            fontWeight: 700,
            letterSpacing: '0.35em',
            marginBottom: Math.round(height * 0.03),
          }}
        >
          PART {index + 1} OF {total}
          {scene.narration ? '  ·  NARRATED' : ''}
        </div>
        <div
          style={{
            color: scene.text_color ?? '#f8fafc',
            fontSize: Math.max(44, Math.round(height * 0.095)),
            fontWeight: 800,
            letterSpacing: '-0.03em',
            lineHeight: 1.08,
            maxWidth: '85%',
          }}
        >
          {scene.title}
        </div>
        <div
          style={{
            backgroundColor: scene.accent_color ?? '#22d3ee',
            height: Math.max(6, Math.round(height * 0.01)),
            margin: `${Math.round(height * 0.03)}px auto 0`,
            width: '16%',
          }}
        />
        <div
          style={{
            color: '#e2e8f0',
            fontSize: Math.max(24, Math.round(height * 0.042)),
            letterSpacing: '0.02em',
            lineHeight: 1.3,
            marginTop: Math.round(height * 0.03),
            maxWidth: '80%',
            opacity: captionIn,
          }}
        >
          {scene.caption}
        </div>
        {scene.visual && (
          <div
            style={{
              color: '#64748b',
              fontSize: Math.max(15, Math.round(height * 0.024)),
              fontStyle: 'italic',
              marginTop: Math.round(height * 0.02),
              opacity: captionIn,
            }}
          >
            {scene.visual}
          </div>
        )}
      </AbsoluteFill>
      {isLast && sources && sources.length > 0 && (
        <div
          style={{
            bottom: '4%',
            color: '#475569',
            fontFamily: 'Arial, sans-serif',
            fontSize: Math.max(13, Math.round(height * 0.02)),
            left: '8%',
            opacity: captionIn,
            position: 'absolute',
            right: '8%',
            textAlign: 'center',
          }}
        >
          Sources: {sources.join('  ·  ').slice(0, 160)}
        </div>
      )}
    </AbsoluteFill>
  );
};

const ProgressBar = () => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  const progress = interpolate(frame, [0, durationInFrames], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return (
    <div
      style={{
        backgroundColor: 'rgba(148, 163, 184, 0.25)',
        bottom: 0,
        height: 8,
        left: 0,
        position: 'absolute',
        right: 0,
      }}
    >
      <div
        style={{
          backgroundColor: '#22d3ee',
          height: '100%',
          width: `${progress * 100}%`,
        }}
      />
    </div>
  );
};

export const SyntheticVideo = ({scenes, sources}: EditDocument) => {
  return (
    <AbsoluteFill>
      {scenes.map((scene, index) => (
        <Sequence
          durationInFrames={scene.duration_frames}
          from={scene.from_frame}
          key={scene.id}
          layout="none"
        >
          <SceneCard
            index={index}
            isLast={index === scenes.length - 1}
            scene={scene}
            sources={sources}
            total={scenes.length}
          />
        </Sequence>
      ))}
      <ProgressBar />
    </AbsoluteFill>
  );
};
