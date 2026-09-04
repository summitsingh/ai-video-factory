declare namespace JSX {
  interface IntrinsicElements {
    [elementName: string]: unknown;
  }
}

declare module 'react/jsx-runtime';
