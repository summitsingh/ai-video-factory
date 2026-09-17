export type EditScene = {
  id: string;
  from_frame: number;
  duration_frames: number;
  title: string;
  caption: string;
  kind?: 'normal' | 'intro' | 'outro';
  visual?: string;
  narration?: string;
  background?: string;
  text_color?: string;
  accent_color?: string;
  image?: string;
  clip?: string;
  subtitle?: string;
  pip?: boolean;
};

export type EditDocument = {
  schema_version: 1;
  width: number;
  height: number;
  fps: number;
  duration_frames: number;
  scenes: EditScene[];
  title?: string;
  description?: string;
  created_at?: string;
  sources?: string[];
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
};

const positiveInteger = (value: unknown, field: string): number => {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new Error(`${field} must be a positive integer`);
  }

  return value as number;
};

const nonNegativeInteger = (value: unknown, field: string): number => {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${field} must be a non-negative integer`);
  }

  return value as number;
};

const nonEmptyString = (value: unknown, field: string): string => {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${field} must be non-empty`);
  }

  return value;
};

const assertExactKeys = (
  value: Record<string, unknown>,
  expected: readonly string[],
  label: string,
): void => {
  const allowed = new Set(expected);
  const unknown = Object.keys(value).find((key) => !allowed.has(key));
  if (unknown !== undefined) {
    throw new Error(`${label} contains unknown field: ${unknown}`);
  }
};

const parseScene = (value: unknown): EditScene => {
  if (!isRecord(value)) {
    throw new Error('scene must be an object');
  }
  assertExactKeys(
    value,
    ['id', 'from_frame', 'duration_frames', 'title', 'caption', 'kind', 'visual', 'narration', 'background', 'text_color', 'accent_color', 'image', 'clip', 'subtitle', 'pip'],
    'scene',
  );

  return {
    id: nonEmptyString(value.id, 'scene id'),
    from_frame: nonNegativeInteger(value.from_frame, 'scene from_frame'),
    duration_frames: positiveInteger(value.duration_frames, 'scene duration_frames'),
    title: nonEmptyString(value.title, 'scene title'),
    caption: nonEmptyString(value.caption, 'scene caption'),
    kind: value.kind === 'normal' || value.kind === 'intro' || value.kind === 'outro'
      ? value.kind
      : 'normal',
    visual: typeof value.visual === 'string' ? value.visual : undefined,
    narration: typeof value.narration === 'string' ? value.narration : undefined,
    background: typeof value.background === 'string' ? value.background : 'gradient',
    text_color: typeof value.text_color === 'string' ? value.text_color : '#f8fafc',
    accent_color: typeof value.accent_color === 'string' ? value.accent_color : '#22d3ee',
    image: typeof value.image === 'string' ? value.image : undefined,
    clip: typeof value.clip === 'string' ? value.clip : undefined,
    subtitle: typeof value.subtitle === 'string' ? value.subtitle : undefined,
    pip: value.pip === true,
  };
};

export const parseEditDocument = (value: unknown): EditDocument => {
  if (!isRecord(value)) {
    throw new Error('edit document must be an object');
  }
  assertExactKeys(
    value,
    ['schema_version', 'width', 'height', 'fps', 'duration_frames', 'scenes', 'title', 'description', 'created_at', 'sources'],
    'edit document',
  );

  if (value.schema_version !== 1) {
    throw new Error('schema_version must be 1');
  }

  if (!Array.isArray(value.scenes)) {
    throw new Error('scenes must be an array');
  }

  const document: EditDocument = {
    schema_version: 1,
    width: positiveInteger(value.width, 'width'),
    height: positiveInteger(value.height, 'height'),
    fps: positiveInteger(value.fps, 'fps'),
    duration_frames: positiveInteger(value.duration_frames, 'duration_frames'),
    scenes: value.scenes.map(parseScene),
    title: typeof value.title === 'string' ? value.title : undefined,
    description: typeof value.description === 'string' ? value.description : undefined,
    created_at: typeof value.created_at === 'string' ? value.created_at : undefined,
    sources: Array.isArray(value.sources) ? value.sources as string[] : undefined,
  };
  const ids = new Set<string>();

  for (const scene of document.scenes) {
    if (ids.has(scene.id)) {
      throw new Error('scene IDs must be unique');
    }
    ids.add(scene.id);

    if (scene.from_frame + scene.duration_frames > document.duration_frames) {
      throw new Error('scene exceeds composition duration');
    }
  }

  return document;
};